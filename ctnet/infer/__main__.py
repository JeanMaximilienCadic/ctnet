

if __name__ == "__main__":
    import torch
    import os
    import argparse
    from torchsummary import summary
    from ai3d.data.dataset import CTDataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from ctnet import CTNet

    parser = argparse.ArgumentParser(description='Process some integers.')
    # ####################################### TRAIN PATH RELATED #########################################################
    parser.add_argument('test_ply_folder'),
    ################################################ HP ################################################################
    parser.add_argument('--dim', default=96, type=int, help="64,96")
    parser.add_argument('--bs', default=10, type=int, help="100,30")
    parser.add_argument('--strech_box', action="store_true")
    parser.add_argument('--no_shuffle', action="store_true")
    parser.add_argument('--dense', action="store_true")
    parser.add_argument('--lr', default=pow(10, -3), type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--t', default=0.9, type=float)
    parser.add_argument('--device', default="cuda", type=str)

    args = parser.parse_args()

    # Model
    model = CTNet(args.dim, id=f"ctnet{args.dim}{'_dense_' if args.dense else '_'}streched_{args.strech_box}").cuda()
    model_path = os.path.realpath(f"__data__/models/{model.id}/{model.id}.pth")
    if args.device =="cuda":
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.cpu()

    # Summary
    if args.device =="cuda":
        summary(model, (args.dim, args.dim, args.dim), batch_size=args.bs)
    else:
        summary(model, (args.dim, args.dim, args.dim), batch_size=args.bs, device="cpu")

    # Loader
    test_dataset = CTDataset(ply_folder=args.test_ply_folder, dim=args.dim)
    # Model variables
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.bs,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             shuffle=not args.no_shuffle,
                             prefetch_factor=5,
                             persistent_workers=True,
                             drop_last=False
                             )
    for x,y, all_ops, ids, xpaths in tqdm(test_loader, desc="Exporting"):
        model.test(x, y, all_ops, ids, xpaths, t=args.t, device=args.device)