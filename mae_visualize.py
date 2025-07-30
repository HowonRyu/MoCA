import os
import numpy as np
import seaborn as sns
import argparse

from sklearn.metrics import confusion_matrix
from mae_visualize_utils import *


from util.datasets import *
import util.misc as misc
from datetime import datetime



def get_args_parser():
    parser = argparse.ArgumentParser('MAE reconstruct', add_help=False)
    parser.add_argument('--task', default="reconstruct", type=str)
    parser.add_argument('--chkpt_dir', default=None, type=str)
    parser.add_argument('--data_path', default='/Users/howonryu/Projects/HAR/mae/data/200', type=str,  # changed
                        help='dataset path')
    parser.add_argument('--data', default='UCIHAR', type=str,
                        help='dataset type')    
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--alt', action='store_true',
                        help='using [n, c, l, 1] format instead') 
    parser.set_defaults(alt=False) 
    parser.add_argument('--RISE_hz', default=30, type=int, 
                        help='RISE data hz')
    parser.add_argument('--nb_classes', default=7, type=int, 
                        help='number of classes')
    parser.add_argument('--batch_size', default=50, type=int, 
                        help='batch size')
    parser.add_argument('--patch_num', default=10, type=int,  
                        help='number of patches per one row in the image')   
    parser.add_argument('--patch_size1', default=1, type=int,  
                        help='images patch size')
    parser.add_argument('--patch_size2', default=20, type=int,  
                        help='images patch size')
    parser.add_argument('--norm_pix_loss', action='store_true') 
    parser.set_defaults(norm_pix_loss=False) 
    parser.add_argument('--LP', action='store_true') 
    parser.set_defaults(LP=False) 
    parser.add_argument('--collapse_labels', action='store_true') 
    parser.set_defaults(collapse_labels=False) 
    parser.add_argument('--finetune_interpolate_patches', action='store_true') 
    parser.set_defaults(finetune_interpolate_patches=False) 
    parser.add_argument('--transform', action='store_true') 
    parser.set_defaults(transform=False) 
    parser.add_argument('--plot_save', action='store_true') 
    parser.set_defaults(plot_save=False)
    parser.add_argument('--plot_save_name', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str, help='output dir path for reconstruction files')
    parser.add_argument('--import_mask_path', default=None, type=str, help='path to import_mask file')
    parser.add_argument('--plot_title', default=None, type=str)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    return parser




def classification(data, data_path, checkpoint_path, patch_size=[1,20], alt=True, nb_classes=7,normalization=False,
                   RISE_hz=30, transform=False, test_data=True, finetune_interpolate_patches=False, LP=False, collapse_labels=False,
                   plot_save_name=None, plot_title=None, labels=None, device='cuda'):
    print("using device: ", device)
    # load data
    patch_num = None
    patch_size1, patch_size2 = patch_size

    dataset, labels, activity_labels, ids, times = load_dataset(data_path=data_path, data=data, test_data=test_data, alt=alt,
                                                            normalization=normalization, RISE_hz=RISE_hz, transform=transform)


    input_size = [dataset[0][0].shape[1], dataset[0][0].shape[2]]

    if int(input_size[1]/patch_size2) !=patch_num:
        patch_num = int(input_size[1]/patch_size2)


    in_chans = dataset[0][0].shape[0]
    print(f"input_size: {input_size}, patch_size:{patch_size}, patch_num: {patch_num}, in_chans: {in_chans}")


    # load model
    print("loading model: ", datetime.now())
    #model_mae_ft = misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    model_mae_ft = load_model_ft(checkpoint_path=checkpoint_path, LP=LP, 
                                    nb_classes=nb_classes, in_chans=in_chans, input_size=input_size, patch_size=patch_size,
                                    finetune_interpolate_patches=finetune_interpolate_patches,
                                    mask_finetune=0, device=device)

    model_mae_ft.to(device)
    print("model loaded: ", datetime.now())

    print("prediction starting: ", datetime.now())
    # predict
    preds_list, targets_list, accuracy = mae_classification_pass(dataset=dataset, model=model_mae_ft, device=device)


    print("prediction finished: ", datetime.now())
    if collapse_labels:
        mapping = {0:0, 1:1, 2:1, 3:1, 4:0, 5:0, 6:0}
        preds_list = [mapping[x] for x in preds_list]
        targets_list = [mapping[x] for x in targets_list]
        labels = ['Sedentary', 'Active']

    # plot the confusion matrix
    cm_test = confusion_matrix(targets_list, preds_list)
    print("saving plot: ", datetime.now())


    plt.figure(figsize=(8, 8))
    sns.heatmap(cm_test, annot=True, cmap='Blues', fmt='d', xticklabels=labels,
                yticklabels=labels, cbar=False, linewidth=.5, annot_kws={"fontsize":16})
    plt.xlabel('Predicted Labels', fontsize=13)
    plt.ylabel('True Labels', fontsize=13)
    plt.title(f'{plot_title}; Accuracy = {accuracy:.3f}')
    plt.show()
    plt.savefig(f'{plot_save_name}_confusion_matrix.png', bbox_inches='tight')
    print("all jobs finished: ", datetime.now())

    

def reconstruct(data_path, data, checkpoint_path, patch_size=[1,20], test_data=True, alt=True,
                mask_ratio=0.75, batch_size = 100, import_mask_path=None, transform=False, norm_pix_loss=False, finetune_interpolate_patches=False,
                normalization=False, device='cuda', output_dir=None):
    print("using device: ", device)
    # load data
    dataset, labels, activity_labels, ids, times = load_dataset(data_path=data_path, data=data, test_data=test_data, alt=alt,
                                                             normalization=normalization, transform=transform)

    patch_num = None
    patch_size1, patch_size2 = patch_size

    input_size = [dataset[0][0].shape[1], dataset[0][0].shape[2]]

    if int(input_size[1]/patch_size2) !=patch_num:
        patch_num = int(input_size[1]/patch_size2)

    in_chans = dataset[0][0].shape[0]

    # load pre_trained model
    model_mae = load_model_pretrain(checkpoint_path=checkpoint_path, input_size=input_size, patch_size=patch_size,
                                    finetune_interpolate_patches=finetune_interpolate_patches,
                                    alt=alt, in_chans=in_chans)
    
    if import_mask_path != None:
        import_mask = torch.load(import_mask_path)
    # reconstruct 
    masked_all, paste_all, x_prime_all, y_all, x_squeeze_all, loss_MSE, loss_nMSE = mae_reconstruction_loop(model=model_mae,
                                                                                                        dataset=dataset, batch_size=batch_size,
                                                                                                        mask_ratio=mask_ratio, alt=alt,
                                                                                                        norm_pix_loss=norm_pix_loss,
                                                                                                        import_mask=import_mask,
                                                                                                        device=device)
    loss_out = torch.concat([loss_MSE, loss_nMSE])
    masked_all.save(f"{output_dir}/masked_all.pt")
    paste_all.save(f"paste_all.pt")
    x_prime_all.save(f"{output_dir}/x_prime_all.pt")
    y_all.save(f"{output_dir}/paste_all.pt")
    x_squeeze_all.save(f"{output_dir}/x_squeeze_all.pt")
    y_all.save(f"{output_dir}/paste_all.pt")
    loss_out.save(f"{output_dir}/loss_out.pt")
    print("all jobs finished: ", datetime.now())
    

   



def main(args):
    if args.task == "reconstruct":
        reconstruct(data=args.data, data_path=args.data_path,  checkpoint_path=args.chkpt_dir, alt=args.alt, device=args.device,
                     patch_size=[args.patch_size1, args.patch_size2], mask_ratio=args.mask_ratio, batch_size = args.batch_size,
                     import_mask_path=args.import_mask_path, normalization=False, RISE_hz=args.RISE_hz,
                     transform=args.transform, test_data=True, finetune_interpolate_patches=args.finetune_interpolate_patches,
                     output_dir=args.output_dir)
    
    if args.task == "classification":        
        classification(data=args.data, data_path=args.data_path, checkpoint_path=args.chkpt_dir, alt=args.alt, device=args.device,
                       patch_size=[args.patch_size1, args.patch_size2], nb_classes=args.nb_classes, collapse_labels=args.collapse_labels,
                       normalization=False, RISE_hz=args.RISE_hz, plot_save_name=args.plot_save_name, plot_title = args.plot_title,
                        transform=args.transform, test_data=True, finetune_interpolate_patches=args.finetune_interpolate_patches, LP=args.LP)



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

