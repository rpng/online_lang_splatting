#!/usr/bin/env python3
"""
Online Language Splatting

This module implements a Simultaneous Localization and Mapping (SLAM) system
based on Gaussian Splatting. It supports RGB-D sensors and language-guided scene understanding.
"""

import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd
from language.load_lang_model import load_lang_model

class SLAM:
    """
    Main RGB-D SLAM system class that integrates frontend, backend, and visualization.
    
    The system follows a standard frontend-backend architecture where:
    - Frontend handles RGB-D frame processing, tracking, and keyframe selection
    - Backend performs mapping and optimization of the 3D Gaussian model
    - Optional GUI provides visualization and parameter adjustment (Note: GUI shows language learning when loaded with language lables from file)
    
    Attributes:
        config: Configuration dictionary loaded from YAML
        save_dir: Directory for saving results and logs
        gaussians: 3D Gaussian model representing the scene
        dataset: Dataset interface for loading RGB-D sensor data
        frontend: SLAM frontend handling tracking and keyframe selection
        backend: SLAM backend handling language learning and mapping and optimization
    """
    def __init__(self, config, save_dir=None):
        """
        Initialize the SLAM system with the given configuration.
        
        Args:
            config: Configuration dictionary loaded from YAML
            save_dir: Directory for saving results and logs (optional)
        """
        # Initialize timing measurement
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # Store configuration
        self.config = config
        self.save_dir = save_dir
        
        # Munchify configuration parameters for convenient dot access
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params = model_params
        self.opt_params = opt_params
        self.pipeline_params = pipeline_params

        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        # Update model parameters
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        # Initialize Gaussian model
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        
        # Load dataset
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        # Load language model if needed
        if config["language"]["language_train"] and not config["language"]["labels_from_file"]:
            t1 = time.time()
            self.lang_model = load_lang_model(model_path=config["language"]["lang_model_path"])
            print("Language model loaded in:{} s".format(time.time() - t1))
        else:
            self.lang_model = None

        # Setup Gaussian model training
        self.gaussians.training_setup(opt_params)
        # Set background color
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        #self.background_language = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        #if config["language"]["language_train"]:
        #    self.background_language = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        #else:
        #    self.background_language = None

        # Initialize communication queues
        frontend_queue = mp.Queue()  # Messages from backend to frontend
        backend_queue = mp.Queue()   # Messages from frontend to backend

        # GUI communication queues (or fake queues if GUI is disabled)
        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config, self.lang_model)

        # Configure frontend
        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        # Configure backend
        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue

        self.backend.set_hyperparams()

        # Setup GUI parameters
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        # Start backend process
        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        # Start backend and frontend
        backend_process.start()
        self.frontend.run()
        backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        # Perform evaluation if enabled
        if self.eval_rendering:
            self._perform_evaluation(frontend_queue, backend_queue)

        # Clean shutdown
        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        
        if self.use_gui and gui_process:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI stopped and joined the main thread")

    def _perform_evaluation(self, frontend_queue: mp.Queue, backend_queue: mp.Queue) -> None:
        """
        Perform evaluation of the online language system
        
        Args:
            frontend_queue: Queue for communication from backend to frontend
            backend_queue: Queue for communication from frontend to backend
        """
        # Get final state from frontend
        self.gaussians = self.frontend.gaussians
        kf_indices = self.frontend.kf_indices
        
        # Evaluate trajectory error
        ATE = eval_ate(
            self.frontend.cameras,
            self.frontend.kf_indices,
            self.save_dir,
            0,
            final=True,
            monocular=False,  # Always use RGB-D mode
        )

        # Evaluate rendering quality before final optimization
        Log("Evaluating rendering quality before final optimization...")
        rendering_result = eval_rendering(
            self.frontend.cameras,
            self.gaussians,
            self.dataset,
            self.save_dir,
            self.pipeline_params,
            self.background,
            kf_indices=kf_indices,
            iteration="before_opt",
            save_images=True,
        )
        
        # Perform color refinement in backend
        Log("Performing final color refinement...")
        while not frontend_queue.empty():
            frontend_queue.get()  # Clear queue
            
        backend_queue.put(["color_refinement"])
        
        # Wait for backend to complete and return refined gaussians
        while True:
            if frontend_queue.empty():
                time.sleep(0.01)
                continue
                
            data = frontend_queue.get()
            if data[0] == "sync_backend" and frontend_queue.empty():
                self.gaussians = data[1]
                break

        # Evaluate rendering quality after final optimization
        Log("Evaluating rendering quality after final optimization...")
        rendering_result = eval_rendering(
            self.frontend.cameras,
            self.gaussians,
            self.dataset,
            self.save_dir,
            self.pipeline_params,
            self.background,
            kf_indices=kf_indices,
            iteration="after_opt",
        )
        
        # Save final model
        save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    
    # Initialize multiprocessing with spawn method for CUDA compatibility
    mp.set_start_method("spawn", force=True)

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running LangGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)

    slam = SLAM(config, save_dir=save_dir)
    #slam.run()

    # All done
    Log("Done.")
