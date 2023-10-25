import argparse
import warnings

def get_args():
    parser = argparse.ArgumentParser(description='TAPS.')
    
    # Basic arguments
    parser.add_argument('--dataset', default='cifar10', help='dataset to use')
    parser.add_argument('--net', required=True, type=str, help='network to use')
    parser.add_argument('--train-batch', default=100, type=int, help='batch size for training')
    parser.add_argument('--test-batch', default=100, type=int, help='batch size for testing')
    parser.add_argument('--n-epochs', default=1, type=int, help='number of epochs')
    parser.add_argument('--load-model', default=None, type=str, help='path of the model to load')
    parser.add_argument('--frac-valid', default=None, type=float, help='fraction of validation samples (none to use no validation)')
    parser.add_argument('--save-dir', default=None, required=False, type=str, help='path to save the logs and the best checkpoint')
    parser.add_argument('--random-seed', default=123, type=int)
    

    # Optimizer and learning rate scheduling
    parser.add_argument('--opt', default='adam', type=str, choices=['adam', 'sgd'], help='optimizer to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD optimizer')
    parser.add_argument('--lr-milestones', default=None,  type=int, nargs='*', help='The milestones for MultiStepLR.')
    parser.add_argument('--lr-factor', default=0.2,  type=float, help='The decay rate of lr.')

    parser.add_argument('--early-stop', action='store_true', help='whether to use early stopping.')
    parser.add_argument('--grad-clip', default=1e10,  type=float)

    # parser.add_argument('--val-check-interval', default=None, type=float, help='frequency of validation.')

    # Configuration of training
    parser.add_argument('--train-eps', default=None,  type=float, help='epsilon to train with')
    parser.add_argument('--test-eps', default=None,  type=float, help='epsilon to test with')
    parser.add_argument('--no-aneal', action='store_true', help='whether to use eps anealling.')
    parser.add_argument('--init', default='default', type=str, choices=['default', 'fast', 'ZerO'], help='Initialization to use')
    parser.add_argument("--grad-accu-batch", default=None, type=int, help="If None, do not use grad accu; if an int, use the specified number as the bs and accumulate grad for the whole batch")
    
    # Configuration of PGD
    parser.add_argument('--step-size', default=None,  type=float, help='the size of each pgd step')
    parser.add_argument('--train-steps', default=None,  type=int, help='the number of pgd steps during training')
    parser.add_argument('--test-steps', default=None,  type=int, help='the number of pgd steps during testing')
    parser.add_argument('--restarts', default=1,  type=int, help='the number of pgd restarts.')
    parser.add_argument('--pgd-weight', default=0.5,  type=float, help='the weight of pgd loss in the training')
    parser.add_argument('--use-adv-training', action='store_true', help='whether to use PGD training. This would ignore any configuration of other training methods, including use-vanilla-ibp.')

    # Configuration of verified training
    # vanilla box training
    parser.add_argument('--cert-weight', default=None,  type=float, help='the weight of certified loss in the training')
    parser.add_argument('--use-vanilla-ibp', action='store_true', help='whether to use vanilla IBP.')
    parser.add_argument('--start-epoch-eps', default=0, type=int)
    parser.add_argument('--end-epoch-eps', default=40, type=int)
    parser.add_argument('--eps-start', default=0, type=float)
    parser.add_argument('--eps-end', default=0, type=float)

    # Small box training (SABR)
    parser.add_argument('--use-small-box', action='store_true', help='whether to use small box. When combined with use-vanilla-ibp, it uses SABR. Otherwise it combines TAPS and SABR.')
    parser.add_argument('--eps-shrinkage', default=1, type=float, help="The effective eps would be shrinkage * eps.")
    parser.add_argument('--relu-shrinkage', default=None, type=float, help="A positive constant smaller than 1, indicating the ratio of box shrinkage after each ReLU.")

    # Configuration of fast regularization
    parser.add_argument('--fast-reg', action='store_true', help='whether to use fast reg.')
    parser.add_argument('--reg-lambda', default=0.5, type=float)
    parser.add_argument('--min-eps-reg', default=1e-6, type=float)



    # TAPS training
    parser.add_argument('--fuse-BN', action='store_true', help='whether to merge BN into parent layers.')

    parser.add_argument('--block-sizes', default=None,  type=int, nargs='*', help='A list of sizes of different blocks. Must sum up to the total number of layers in the network.')
    # parser.add_argument('--verify-method', default='DeepZ',  type=str, choices=['DeepZ'], help='The verifying method. Used to determine the combination coefficient of PGD and Box bounds.')
    parser.add_argument('--pgd-weight-start', default=1,  type=float, help='the start value of the weight of the pgd bounds')
    parser.add_argument('--pgd-weight-end', default=1,  type=float, help='the end value of the weight of the pgd bounds')
    parser.add_argument('--start-epoch-pgd-weight', default=0,  type=int)
    parser.add_argument('--end-epoch-pgd-weight', default=0,  type=int)
    parser.add_argument('--estimation-batch', default=None, type=int, help='batch size for bound estimation.')
    # parser.add_argument('--num-pivotal', type=int, help='Number of pivotal points for bound estimation. Double #pivotal points is used as we need to estimate both lower bound and higher bound.')
    parser.add_argument('--L1-reg', default=0,  type=float, help='the L1 reg coefficient for the last block.')
    parser.add_argument('--L2-reg', default=0, type=float, help='the L2 reg coefficient for the last block.')
    # parser.add_argument('--kappa', default=0, type=float, help='the confidence parameter of PGD.')
    # parser.add_argument('--volreg-lambda', default=0.5, type=float, help='the coefficient of Box reg.')
    parser.add_argument('--layers-after-flatten-to-fuse', default=0, type=float, help='the number of layers (using flatten layer as baseline) for fusing the BN layers.')
    parser.add_argument('--soft-thre', default=0.5, type=float, help='the hyperparameter of soft gradient link.')
    parser.add_argument('--min-eps-pgd', default=0, type=float, help='the min eps when using PGD.')
    parser.add_argument('--alpha-box', default=1, type=float, help='the exponential coef of box loss in the mix training.')
    parser.add_argument('--no-ibp-anneal', action='store_true', help='whether to use ibp for annealing.')
    parser.add_argument('--no-ibp-reg', action='store_true', help='whether to use multiplying IBP loss in the backward.')
    parser.add_argument('--use-single-estimator', action='store_true', help='whether to use single-estimator PGD instead of multi-estimator.')




    # certify
    parser.add_argument('--mode', default=None, required=False, type=str, choices=['box_trained', 'mix_trained', 'certify'], help='Indicates models from which mode should be certified.')
    parser.add_argument('--load-certify-file', default=None, type=str, help='the certify file to load. A single filename in the same directory as the model.')
    parser.add_argument('--mnbab-config', default=None, type=str, help='the config file for MN-BaB.')
    parser.add_argument('--tolerate-error', action='store_true', help='Whether to ignore MNBaB errors. Normally these are memory overflows.')
    parser.add_argument('--start-idx', default=None, type=int, help='the start index of the input in the test dataset (inclusive).')
    parser.add_argument('--end-idx', default=None, type=int, help='the end index of the input in the test dataset (exclusive).')



    # Metadata
    parser.add_argument('--root-dir', required=False, default='./', type=str, help='directory to store the data')

    args = parser.parse_args()


    # check training parameters
    if not args.mode == "certify":
        assert 0 <= args.cert_weight <= 1, "Cert weight must be between 0 and 1."
        if args.block_sizes is None:
            print("-----No block-sizes is provided. End-to-end training will be used.----")
        if args.train_eps is None:
            raise ValueError("Must specify the train-eps.")
        if args.test_eps is None:
            args.test_eps = args.train_eps
        if args.eps_end == 0:
            args.eps_end = args.train_eps
        if args.estimation_batch is None:
            args.estimation_batch = args.train_batch
        if args.relu_shrinkage is not None:
            assert 0 <= args.relu_shrinkage < 1
    else:
        assert args.test_eps is not None, "A test eps is required for certification."
        assert args.load_model is not None, "A saved model is required to be loaded."
        # if not args.use_vanilla_ibp:
        #     assert args.block_sizes is not None, "Should provide two integers for block sizes"
        # assert args.block_sizes is not None, "Certify requires to split the net blockwisely. If you want a single block, please pass in a single integer, i.e., the total number of layers. Careful: using a single integer means to use MILP for the whole network, which is costly."
        # assert args.cert_weight is not None, "A certification loss weight should be specified."
        # assert args.pgd_weight_for_comb is not None, "The cert mode uses PGD+Box for certification. A PGD weight should be specified."
            
    return args
