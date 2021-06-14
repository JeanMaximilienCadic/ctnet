import torch

if __name__ == "__main__":
    import argparse
    from torchsummary import summary
    from ctdataset.dataset import CTDataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from ctnet import CTNet
    from torch.nn import BCELoss
    from torch.optim import Adam
    from ctnet.trainer import f1_score
    import torch
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
    parser.add_argument('--no_caching', action="store_true")

    args = parser.parse_args()

    # Model
    model = CTNet(args.dim, id=f"ctnet{args.dim}{'_dense_' if args.dense else '_'}streched_{args.strech_box}").cuda()
    model.to(args.device)

    # Summary
    if args.device =="cuda":
        summary(model, (args.dim, args.dim, args.dim), batch_size=args.bs)
    else:
        summary(model, (args.dim, args.dim, args.dim), batch_size=args.bs, device="cpu")

    # Loader
    test_dataset = CTDataset(ply_folder=args.test_ply_folder,
                             dim=args.dim,
                             caching=not args.no_caching)
    # Model variables
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.bs,
                             num_workers=args.num_workers,
                             shuffle=not args.no_shuffle,
                             drop_last=False
                             )

    optimizer = Adam(model.parameters(), lr=pow(10,-4))
    criterion = BCELoss()
    mu_score = 0
    for epoch in range(100):
        scores = torch.Tensor(len(test_loader)).to(args.device)
        for k, (x,y, all_ops, ids, xpaths) in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Epoch {epoch} >> {mu_score}"):
            optimizer.zero_grad()
            x, y = x.cuda(), y.cuda()
            _y = model(x)
            loss = criterion(_y, y)
            loss = loss / _y.size(1)  # average the loss by minibatch
            loss.backward()
            loss.item()
            scores[k] = f1_score(y, _y)
            optimizer.step()
        mu_score = scores.mean()