import parameters
from data import get_dataloader
import routines
import baseline
import wasserstein_ensemble
import os
import utils
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import json
from model import get_model_from_name
from routines import test  
PATH_TO_CIFAR = "./cifar/"
sys.path.append(PATH_TO_CIFAR)
import train as cifar_train
from tensorboardX import SummaryWriter
from copy import deepcopy
import collections
import torch.optim as optim
from routines import train
import math
#=================================== SEED ========================================

INIT_SEED=[1,2,3,4,5,6,7,8]

#=================================== SEED ========================================

def extract_mask(model_dict):

    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = deepcopy(model_dict[key])

    return new_dict

#def combine_mask(model_dict,mask_dict):
#    with torch.no_grad():
#        for name,m in model.named_modules():
 #           if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                mask_name = name+'.weight_mask'
#                if mask_name in mask_dict.keys():
#                    eval(f"model.{name}.weight_mask").set_(mask_dict[mask_name])
 #               else:
#                    print('Can not find [{}] in mask_dict'.format(mask_name))
 #   return new_dict

def check_zero(model_dict):
    print(model_dict.keys())
    for key, value in model_dict.items():
        print(f"the ratio of zero for {key}")
        print(1-torch.count_nonzero(value)/value.numel())
    
def stat_zero(mask_dict):
    zero={}
    for key in mask_dict.keys():
        value=mask_dict[key]
        zero[key]=1-torch.count_nonzero(value)/value.numel()    
    return zero

def pruning_model_local(model, px):
    #print('Apply Unstructured L1 Pruning Globally (all conv layers)')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            prune.l1_unstructured(m, 'weight', amount=px)

            
def pruning_model_local_layer(model, px_dict):
    #print('Apply Unstructured L1 Pruning Globally (all conv layers)')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        mask_name = name+'.weight_mask'
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            prune.l1_unstructured(m, 'weight', amount=px_dict[mask_name])         
            
            
def pruning_model_global(model, px):
    #print('Apply Unstructured L1 Pruning Globally (all conv layers)')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            parameters_to_prune.append((m,'weight'))
    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def prune_model_custom(model, mask_dict):

    #print('Pruning with custom mask (all conv layers)')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            mask_name = name+'.weight_mask'
            if mask_name in mask_dict.keys():
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            else:
                print('Can not find [{}] in mask_dict'.format(mask_name))

def prune_model_remove(model):

    #print('Pruning with custom mask (all conv layers)')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.utils.prune.remove(m, name='weight')

                
def make_dir(dirs):  
    if not os.path.exists(dirs):
        os.makedirs(dirs)

if __name__ == '__main__':

    print("------- Setting up parameters -------")
    args = parameters.get_parameters()
    print("The parameters are: \n", args)

    if args.deterministic:
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    ######################################################################
    # add by us
    ######################################################################
    experiment_dir=args.experiment_dir
    prunint_rate=args.prunint_rate
    run_mode=args.run_mode
    
    
    # loading configuration
    config, second_config = utils._get_config(args)
    args.config = config
    args.second_config = second_config
        
    if run_mode in ['finetune','lottery','prune','prune_times']:
        train_loader, test_loader = get_dataloader(args)
        retrain_loader, _ = get_dataloader(args, no_randomness=args.no_random_trainloaders)
        models = []
        masks = []
        accuracies = []
        for i in range(args.num_models):
            torch.backends.cudnn.benchmark = True
            torch.manual_seed(INIT_SEED[i])
            model=get_model_from_name(args, i)
            mask_model=get_model_from_name(args, i)
            if(run_mode in ['finetune','prune','prune_times']):
                model.load_state_dict(torch.load(f'./{experiment_dir}/origin_model/model{i}.model'))
            
            pre_mask_model_dict=torch.load(f'./{experiment_dir}/origin_mask/model{i}.model')            
            """mask_model_dict=collections.OrderedDict()            
            for key in pre_mask_model_dict.keys():
                if(key[-5::]=='_orig'):
                    mask_model_dict[key[0:-5]]=pre_mask_model_dict[key]
                else:
                    mask_model_dict[key]=pre_mask_model_dict[key]"""
            pruning_model_local(mask_model,args.prunint_rate)
            mask_model.load_state_dict(pre_mask_model_dict)
            mask = extract_mask(mask_model.state_dict())
            models.append(model)
            masks.append(mask)
        with open(f"./{experiment_dir}/acc.json",'r') as load_f:
            accuracies= json.load(load_f)["accuracies"]
            
        if(run_mode == 'prune'):
            accuracies=[]
            losses=[]
            for i in range(args.num_models):
                prune_model_custom(models[i],masks[i])
                #print("========================================================")
                #print("=========================================================")
                #print("did i prune yet?")
                log_dict = {}
                log_dict['train_losses'] = []
                log_dict['train_counter'] = []
                log_dict['test_losses'] = []
                acc,loss = test(args, models[i], test_loader, log_dict,return_loss=True)
                accuracies.append(acc)
                losses.append(loss)
                prune_model_remove(models[i])
                #print("========================================================")
                #print("accuracies",accuracies)
                #print("=========================================================")
                #checking
                
                #print("========================================================")
                #print("=========================================================")
                #print("the status of models")
                
        elif(run_mode in ['lottery', 'finetune']):
            accuracies=[]
            losses=[]
            for i in range(args.num_models):
                prune_model_custom(models[i],masks[i])
                optimizer = optim.SGD(models[i].parameters(), lr=args.learning_rate,
                          momentum=args.momentum)
                log_dict = {}
                log_dict['train_losses'] = []
                log_dict['train_counter'] = []
                log_dict['test_losses'] = []
                for epoch in range(1, args.n_epochs + 1):
                    train(args, models[i], optimizer, train_loader, log_dict, epoch, model_id=str(i))
                    acc,loss= test(args, models[i], test_loader, log_dict,return_loss=True)
                accuracies.append(acc)
                losses.append(loss)
                prune_model_remove(models[i])
                
        elif(run_mode == 'prune_times'):
            prune_times=args.prune_times
            ratio=args.prunint_rate/prune_times
            cumm_ratio=ratio
            print(f'enter prune_times, the times is {prune_times},ratio:{ratio}')
            for j in range(prune_times):
                accuracies=[]
                losses=[]

                for i in range(args.num_models):
                    pruning_model_local(models[i],cumm_ratio)
                    optimizer = optim.SGD(models[i].parameters(), lr=args.learning_rate,
                              momentum=args.momentum)
                    log_dict = {}
                    log_dict['train_losses'] = []
                    log_dict['train_counter'] = []
                    log_dict['test_losses'] = []
                    for epoch in range(1, args.n_epochs + 1):
                        train(args, models[i], optimizer, train_loader, log_dict, epoch, model_id=str(i))
                        acc,loss= test(args, models[i], test_loader, log_dict,return_loss=True)
                    accuracies.append(acc)
                    losses.append(loss)
                    prune_model_remove(models[i])
                print(f"in times {j}, now cumm_ratio {cumm_ratio},accuracies:{accuracies},losses:{losses}")
                cumm_ratio+=ratio
        else:
            raise "Wrong model"
        #print("========================================================")
        #print("accuracies",accuracies)
        #print("=========================================================")
    elif run_mode== 'benchmark'or run_mode== 'benchmark_cross_mask':
        finetune_time = args.finetunetimes_benchmark
        # get dataloaders
        print("------- Obtain dataloaders -------")
        train_loader, test_loader = get_dataloader(args)
        retrain_loader, _ = get_dataloader(args, no_randomness=args.no_random_trainloaders)
        
        print("------- Training independent models -------")
        #models, accuracies = routines.train_models(args, train_loader, test_loader,seed=INIT_SEED)
        accuracies,models,losses=[],[],[]
        for idx in range(args.num_models):
            if args.dataset.lower()[0:7] == 'cifar10' and (args.model_name.lower()[0:5] == 'vgg11' or args.model_name.lower()[0:6] == 'resnet'):
                if idx == 0:
                    config_used = config
                elif idx == 1:
                    config_used = second_config
# <<<<<<< HEAD
                print("QWERTY: Enter Cifar 1")
# =======      
# >>>>>>> 702e4b159e4964754752a07e6e70f80b5c84b67d
                model, accuracy = cifar_train.get_pretrained_model(
                    config_used, os.path.join(args.load_model_dir, 'model_{}/{}.checkpoint'.format(idx, args.ckpt_type)),
                    args.gpu_id, relu_inplace=not args.prelu_acts # if you want pre-relu acts, set relu_inplace to False
                )
    
            else:
                model, accuracy = routines.get_pretrained_model(
                    args, os.path.join(args.load_model_dir, 'model_{}/{}.checkpoint'.format(idx, args.ckpt_type)), idx = idx
                )
            log_dict = {}
            log_dict['train_losses'] = []
            log_dict['train_counter'] = []
            log_dict['test_losses'] = []
            models.append(model)
            acc,loss= test(args, models[idx], test_loader, log_dict,return_loss=True)
            print("He says accury is ",accuracy,"but actually",acc)
            accuracies.append(acc)
            losses.append(loss)
        nicks=[f'model_{i}' for i in range(len(models))]
        #print('num_epochs---------------++++++',config['num_epochs'])
# <<<<<<< HEAD
        models, accuracy=routines.train_models(args, retrain_loader, test_loader, re_models=models)
        #print("models",models)    
# =======
#         models, accuracy=routines.retrain_models(args, models, retrain_loader, test_loader, config,initial_acc=accuracies,nicks=nicks)
            
# >>>>>>> 702e4b159e4964754752a07e6e70f80b5c84b67d
            
        if(run_mode== 'benchmark'):
            for i in range(len(models)):            
                model=models[i]
                make_dir(f"./{experiment_dir}/origin_model")
    # <<<<<<< HEAD
                torch.save(model.state_dict(), f'./{experiment_dir}/origin_model/model{i}.model')
                model_tem=deepcopy(model)
                pruning_model_local(model_tem,args.prunint_rate)
                make_dir(f"./{experiment_dir}/origin_mask")
                torch.save(model_tem.state_dict(), f'./{experiment_dir}/origin_mask/model{i}.model')
    # =======
    #             torch.save(model.state_dict(), f'./model{i}.model')
    #             model_tem=deepcopy(model)
    #             pruning_model_local(model_tem,args.prunint_rate)
    #             make_dir(f"./{experiment_dir}/origin_mask")
    #             torch.save(model.state_dict(), f'/model{i}.model')
    # >>>>>>> 702e4b159e4964754752a07e6>e70f80b5c84b67d
        elif(run_mode == 'benchmark_cross_mask'):
            mask_models=[]
            for i in range(len(models)):            
                model=models[i]
                make_dir(f"./{experiment_dir}/origin_model")
                torch.save(model.state_dict(), f'./{experiment_dir}/origin_model/model{i}.model')
                model_tem=deepcopy(model)
                pruning_model_local(model_tem,args.prunint_rate)
                mask_models.append(model_tem)
                
            assert(len(mask_models)==2,"Now only valid for 2 models")
            mask0,mask1=extract_mask(mask_models[0].state_dict()),extract_mask(mask_models[1].state_dict())
            both_mask={}
            for key in mask0.keys():
                both_mask[key]=mask0[key]
                both_mask[key][mask1[key] == 1]= 1
            model_tem0,model_tem1=deepcopy(models[0]),deepcopy(models[1])
            prune_model_custom(model_tem0,both_mask)
            prune_model_custom(model_tem1,both_mask)
            #prune_model_remove(model_tem0)
            #prune_model_remove(model_tem1)
            both_zero_rate=stat_zero(both_mask)
            print("both_zero_rate",both_zero_rate)
            """
            both_zero_rate=stat_zero(both_mask)
            new_zero_rate={}
            print("both_zero_rate",both_zero_rate)
            for key in both_zero_rate.keys():
                new_zero_rate[key]=((args.prunint_rate-both_zero_rate[key])/2+both_zero_rate[key]).item()
            print("new_zero_rate",new_zero_rate)
                    
            pruning_model_local_layer(model_tem0,new_zero_rate)
            pruning_model_local_layer(model_tem1,new_zero_rate)
            mask0,mask1=extract_mask(model_tem0.state_dict()),extract_mask(model_tem1.state_dict())
            final_mask={}
            for key in mask0.keys():
                final_mask[key]=mask0[key]
                final_mask[key][mask1[key] == 0]= 0
            
            model_tem0,model_tem1=deepcopy(models[0]),deepcopy(models[1])
            prune_model_custom(model_tem0,final_mask)
            prune_model_custom(model_tem1,final_mask)
            print("===========================================")
            print("===========================================")
            print("Check final_mask")
            print("===========================================")
            print("===========================================")
            check_zero(final_mask)
            """
            mask_models=[model_tem0,model_tem1]
            
            for i in range(len(mask_models)):
                make_dir(f"./{experiment_dir}/origin_mask")
                torch.save(mask_models[i].state_dict(), f'./{experiment_dir}/origin_mask/model{i}.model')
            
            
        with open(f"./{experiment_dir}/acc.json","w") as f:
            json.dump({
                "accuracies":accuracies,
                "losses":losses
            },f)
            
    else:
        raise "wrong mode"
    
    print("===============================================")
    print("===============================================")
    print("before Ensemble,acc:",accuracies,"loss",losses)
    print("===============================================")
    print("===============================================")
    
    for i in range(len(models)):
        print(f"check zero rate for model {i}")
        check_zero(models[i].state_dict())
        
    # if args.debug:
    #     print(list(models[0].parameters()))

    if args.same_model!=-1:
        print("Debugging with same model")
        model, acc = models[args.same_model], accuracies[args.same_model]
        models = [model, model]
        accuracies = [acc, acc]

    for name, param in models[0].named_parameters():
        print(f'layer {name} has #params ', param.numel())

    import time
    # second_config is not needed here as well, since it's just used for the dataloader!
    print("Activation Timer start")
    st_time = time.perf_counter()
    activations = utils.get_model_activations(args, models, config=config)
    end_time = time.perf_counter()
    setattr(args, 'activation_time', end_time - st_time)
    print("Activation Timer ends")

    for idx, model in enumerate(models):
        setattr(args, f'params_model_{idx}', utils.get_model_size(model))

    # if args.ensemble_iter == 1:
    #
    # else:
    #     # else just recompute activations inside the method iteratively
    #     activations = None


    # set seed for numpy based calculations
    NUMPY_SEED = 100
    np.random.seed(NUMPY_SEED)

    # run geometric aka wasserstein ensembling
    print("------- Geometric Ensembling -------")
    # Deprecated: wasserstein_ensemble.geometric_ensembling(models, train_loader, test_loader)


    print("Timer start")
    st_time = time.perf_counter()
    
       ############################################
    # md_dict = models[0].state_dict()
    # check_zero(md_dict)
        ## 這裏放非零百分比計算函數,循環dict的weight  state_dict()
        ######################
        ###########################################
        
    geometric_acc, geometric_model,geometric_loss = wasserstein_ensemble.geometric_ensembling_modularized(args, models, train_loader, test_loader, activations,return_loss=True)
    print("===============================================")
    print("===============================================")
    print("After Ensemble,acc:",geometric_acc,"loss",geometric_loss)
    print("===============================================")
    print("===============================================")
    end_time = time.perf_counter()
    print("Timer ends")
    setattr(args, 'geometric_time', end_time - st_time)
    args.params_geometric = utils.get_model_size(geometric_model)

    print("Time taken for geometric ensembling is {} seconds".format(str(end_time - st_time)))
    # run baselines
    print("------- Prediction based ensembling -------")
    prediction_acc = baseline.prediction_ensembling(args, models, test_loader)

    print("------- Naive ensembling of weights -------")
    naive_acc, naive_model = baseline.naive_ensembling(args, models, test_loader)

    if args.retrain > 0:
        print('-------- Retraining the models ---------')
        if args.tensorboard:
            tensorboard_dir = os.path.join(args.tensorboard_root, args.exp_name)
            utils.mkdir(tensorboard_dir)
            print("Tensorboard experiment directory: {}".format(tensorboard_dir))
            tensorboard_obj = SummaryWriter(log_dir=tensorboard_dir)
        else:
            tensorboard_obj = None

        if args.retrain_avg_only or args.retrain_geometric_only:
            if args.retrain_geometric_only:
                initial_acc = [geometric_acc]
                nicks = ['geometric']
                _, best_retrain_acc = routines.retrain_models(args, [geometric_model], retrain_loader,
                                                              test_loader, config, tensorboard_obj=tensorboard_obj,
                                                              initial_acc=initial_acc, nicks=nicks)
                args.retrain_geometric_best = best_retrain_acc[0]
                args.retrain_naive_best = -1
            else:
                if naive_acc < 0:
                    initial_acc = [geometric_acc]
                    nicks = ['geometric']
                    _, best_retrain_acc = routines.retrain_models(args, [geometric_model],
                                                                  retrain_loader, test_loader, config,
                                                                  tensorboard_obj=tensorboard_obj,
                                                                  initial_acc=initial_acc, nicks=nicks)
                    args.retrain_geometric_best = best_retrain_acc[0]
                    args.retrain_naive_best = -1
                else:
                    nicks = ['geometric', 'naive_averaging']
                    initial_acc = [geometric_acc, naive_acc]
                    _, best_retrain_acc = routines.retrain_models(args, [geometric_model, naive_model], retrain_loader, test_loader, config, tensorboard_obj=tensorboard_obj, initial_acc=initial_acc, nicks=nicks)
                    args.retrain_geometric_best = best_retrain_acc[0]
                    args.retrain_naive_best = best_retrain_acc[1]

            args.retrain_model0_best = -1
            args.retrain_model1_best = -1

        else:

            if args.skip_retrain == 0:
                original_models = [models[1]]
                original_nicks = ['model_1']
                original_accuracies = [accuracies[1]]
            elif args.skip_retrain == 1:
                original_models = [models[0]]
                original_nicks = ['model_0']
                original_accuracies = [accuracies[0]]
            elif args.skip_retrain < 0:
                original_models = models
                original_nicks = ['model_0', 'model_1']
                original_accuracies = accuracies
            else:
                raise NotImplementedError

            if naive_acc < 0:
                # this happens in case the two models have different layer sizes
                nicks = original_nicks + ['geometric']
                initial_acc = original_accuracies + [geometric_acc]
                _, best_retrain_acc = routines.retrain_models(args, [*original_models, geometric_model],
                                                              retrain_loader, test_loader, config,
                                                              tensorboard_obj=tensorboard_obj, initial_acc=initial_acc, nicks=nicks)
                args.retrain_naive_best = -1
            else:
                nicks = original_nicks + ['geometric', 'naive_averaging']
                initial_acc = [*original_accuracies, geometric_acc, naive_acc]
                _, best_retrain_acc = routines.retrain_models(args, [*original_models, geometric_model, naive_model], retrain_loader, test_loader, config, tensorboard_obj=tensorboard_obj, initial_acc=initial_acc, nicks=nicks)
                args.retrain_naive_best = best_retrain_acc[len(initial_acc)-1]

            if args.skip_retrain == 0:
                args.retrain_model0_best = -1
                args.retrain_model1_best = best_retrain_acc[0]
            elif args.skip_retrain == 1:
                args.retrain_model0_best = best_retrain_acc[0]
                args.retrain_model1_best = -1
            elif args.skip_retrain < 0:
                args.retrain_model0_best = best_retrain_acc[0]
                args.retrain_model1_best = best_retrain_acc[1]
            
            args.retrain_geometric_best = best_retrain_acc[len(original_models)]

    if args.save_result_file != '':

        results_dic = {}
        results_dic['exp_name'] = args.exp_name

        for idx, acc in enumerate(accuracies):
            results_dic['model{}_acc'.format(idx)] = acc

        results_dic['geometric_acc'] = geometric_acc
        results_dic['prediction_acc'] = prediction_acc
        results_dic['naive_acc'] = naive_acc

        # Additional statistics
        results_dic['geometric_gain'] = geometric_acc - max(accuracies)
        results_dic['geometric_gain_%'] = ((geometric_acc - max(accuracies))*100.0)/max(accuracies)
        results_dic['prediction_gain'] = prediction_acc - max(accuracies)
        results_dic['prediction_gain_%'] = ((prediction_acc - max(accuracies)) * 100.0) / max(accuracies)
        results_dic['relative_loss_wrt_prediction'] = results_dic['prediction_gain_%'] - results_dic['geometric_gain_%']

        if args.eval_aligned:
            results_dic['model0_aligned'] = args.model0_aligned_acc

        results_dic['geometric_time'] = args.geometric_time
        # Save retrain statistics!
        if args.retrain > 0:
            results_dic['retrain_geometric_best'] = args.retrain_geometric_best * 100
            results_dic['retrain_naive_best'] = args.retrain_naive_best * 100
            if not args.retrain_avg_only:
                results_dic['retrain_model0_best'] = args.retrain_model0_best * 100
                results_dic['retrain_model1_best'] = args.retrain_model1_best * 100
            results_dic['retrain_epochs'] = args.retrain

        utils.save_results_params_csv(
            os.path.join(args.csv_dir, args.save_result_file),
            results_dic,
            args
        )

        print('----- Saved results at {} ------'.format(args.save_result_file))
        print(results_dic)


    print("FYI: the parameters were: \n", args)
