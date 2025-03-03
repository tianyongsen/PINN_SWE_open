import argparse
import time
import os
from trainer import Trainer


os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import torch
import deepxde as dde

from src.model.laaf import DNN_GAAF, DNN_LAAF
from src.optimizer import MultiAdam, LR_Adaptor, LR_Adaptor_NTK, Adam_LBFGS
from src.pde.swe import SWE2D
from src.utils.args import parse_hidden_layers, parse_loss_weight
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback
from src.utils.rar import rar_wrapper

from cases.static_cases import  Static_Flat_Case,\
                                Static_Cosine_Bump_Case,\
                                Static_Cosine_Depression_Case,\
                                Static_Step_Case, \
                                Static_Tidal_Case,\
                                Static_Flat_Rain_Case,\
                                Static_Cosine_Bump_Rain_Case,\
                                Static_Cosine_Depression_Rain_Case,\
                                Static_Step_Rain_Case,\
                                Static_Tidal_Rain_Case   #10 cases

from cases.dynamic_cases import Dynamic_Dam_Break_Var_Cons_Case,\
                                Dynamic_Dam_Break_Var_Cons_Entropy_Case,\
                                Dynamic_Dam_Break_Primitive_Var_Case,\
                                Dynamic_Tidal_Var_Case,\
                                Dynamic_Tidal_Var_Rain_Case,\
                                Dynamic_Circular_Dam_Break_Var_Case  #5 cases

cases_list_static=[(SWE2D,Static_Flat_Case()),
           (SWE2D,Static_Cosine_Bump_Case()),
           (SWE2D,Static_Cosine_Depression_Case()),
           (SWE2D,Static_Step_Case()),
           (SWE2D,Static_Tidal_Case()),
           (SWE2D,Static_Flat_Rain_Case()),
           (SWE2D,Static_Cosine_Bump_Rain_Case()),
           (SWE2D,Static_Cosine_Depression_Rain_Case()),
           (SWE2D,Static_Step_Rain_Case()),
           (SWE2D,Static_Tidal_Rain_Case())
           ]
cases_list_dynamic=[(SWE2D,Dynamic_Dam_Break_Var_Cons_Case()),
                   (SWE2D,Dynamic_Dam_Break_Var_Cons_Entropy_Case()),
                    (SWE2D,Dynamic_Dam_Break_Primitive_Var_Case()),
                   (SWE2D,Dynamic_Tidal_Var_Case()),
                   (SWE2D,Dynamic_Tidal_Var_Rain_Case()),
                   (SWE2D,Dynamic_Circular_Dam_Break_Var_Case())
                    ]

cases_list=cases_list_static+cases_list_dynamic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINN SWE')
    parser.add_argument('--name', type=str, default="ALL_CASES")
    parser.add_argument('--device', type=str, default="cpu")  # set to "cpu" enables cpu training and set "0" for gpu training
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--hidden-layers', type=str, default='300*6')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--iter', type=int, default=50000)
    parser.add_argument('--log-every', type=int, default=1000)  #log(日志) every 100 steps
    parser.add_argument('--plot-every', type=int, default=1000) #plot every 2000 steps
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--method', type=str, default="laaf")
    parser.add_argument('--decay',type=list,default=['step',30000,1e-5/1e-3])

    command_args = parser.parse_args()

    seed = command_args.seed
    if seed is not None:
        dde.config.set_random_seed(seed)
    date_str = time.strftime('%m.%d-%H.%M.%S', time.localtime())
    trainer = Trainer(f"{date_str}-{command_args.name}", command_args.device)

    for case_config in cases_list:

        def get_model_dde():
            swe_case = case_config[0](**case_config[1]())   #an instance of SWE problem

            if command_args.method == "gepinn":
                swe_case.use_gepinn()

            #network
            net = dde.nn.FNN([swe_case.input_dim] + parse_hidden_layers(command_args) + [swe_case.output_dim], 
                             "tanh", 
                             "Glorot normal",
                             )
            
            # net.regularizer =['l2',0.001]    #tianyongfu ,l2 regularization

            if command_args.method == "laaf":
                #attention: the layyers substract 1 
                net = DNN_LAAF(len(parse_hidden_layers(command_args)) - 1, parse_hidden_layers(command_args)[0], swe_case.input_dim, swe_case.output_dim)
            elif command_args.method == "gaaf":
                net = DNN_GAAF(len(parse_hidden_layers(command_args)) - 1, parse_hidden_layers(command_args)[0], swe_case.input_dim, swe_case.output_dim)
            net = net.float()    # convert to float precision

            opt = torch.optim.Adam(net.parameters(), command_args.lr)
            # from torch.optim.lr_scheduler import StepLR
            # scheduler = StepLR(opt, step_size=100, gamma=1e-3)

            if command_args.method == "multiadam":
                opt = MultiAdam(net.parameters(), lr=1e-3, betas=(0.99, 0.99), loss_group_idx=[swe_case.num_pde])
            elif command_args.method == "lra":
                opt = LR_Adaptor(opt, swe_case.loss_weights, swe_case.num_pde)
            elif command_args.method == "ntk":
                opt = LR_Adaptor_NTK(opt, swe_case.loss_weights, swe_case.num_pde)
            elif command_args.method == "lbfgs":
                opt = Adam_LBFGS(net.parameters(), switch_epoch=5000, adam_param={'lr':command_args.lr})


            model = swe_case.create_model(net)               # create the PINN model
            model.compile(opt, loss_weights=swe_case.loss_weights,decay=command_args.decay)    # compile the model with optimizer and loss weights
            if command_args.method == "rar":
                model.train = rar_wrapper(swe_case, model, {"interval": 1000, "count": 1})
            # the trainer calls model.train(**train_args)
            return model

        def get_model_others():
            model = None
            # create a model object which support .train() method, and param @model_save_path is required
            # create the object based on command_args and return it to be trained
            # schedule the task using trainer.add_task(get_model_other, {training args})
            return model

        trainer.add_task(
            get_model_dde, {
                "iterations": command_args.iter,
                "display_every": command_args.log_every,
                "callbacks": [
                    TesterCallback(log_every=command_args.log_every),
                    PlotCallback(log_every=command_args.plot_every, fast=True),
                    LossCallback(verbose=True),
                ]
            }
        )

    trainer.setup(__file__, seed)
    trainer.set_repeat(command_args.repeat)
    trainer.train_all()
    trainer.summary()
