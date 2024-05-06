from Model import CUM

def parse_method(args, dataset, n, c, d, device):
    if args.method == 'CUMnoUIL':
        model=CUM(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout,
                    num_heads=args.num_heads, use_bn=args.use_bn, nb_random_features=args.M,
                    use_gumbel=args.use_gumbel, use_residual=args.use_residual, use_act=args.use_act, use_jk=args.use_jk,
                    nb_gumbel_sample=args.K, rb_order=args.rb_order, rb_trans=args.rb_trans, filter1=args.filter1, filter2=args.filter2).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):

    parser.add_argument('--method', '-m', type=str, default='CUMnoUIL')
    parser.add_argument('--dataset', type=str, default='crosssite')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets with fixed splits, semi or supervised')
    parser.add_argument('--rand_split', action='store_true',default=True, help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20, help='labeled nodes randomly selected')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                        help='evaluation metric')
    parser.add_argument('--save_init_edge', action='store_true', help='whether to save the edge information obtained by initialization')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--save_index', type=str, default='', help='Serial number when saving files')
    parser.add_argument('--model_dir', type=str, default='model/')
    parser.add_argument('--logger_file',type=str,default='results/')

    # hyper-parameter for model training
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')

    # hyper-parameter for UserEncoder
    parser.add_argument('--filter1', type=int, default=1)
    parser.add_argument('--filter2', type=int, default=2)

    # hyper-parameter for CrossModuleStage1
    parser.add_argument('--init_edge_method', type=str, default='ks', help='Method of initializing edge information', choices = ['knn','scorer','ks'])
    parser.add_argument('--scorer_type',type=str, default='RF', help='Types of scorer models', choices = ['MLP','RF'])
    parser.add_argument('--knn_num', type=int, default=5, help='number of k for KNN graph')
    parser.add_argument('--score_thold',type=float,default=0.5,help='threshold of scorer for init edge')

    # hyper-parameter for CrossModuleStage2
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--M', type=int,
                        default=30, help='number of random features')
    parser.add_argument('--use_gumbel', action='store_true', help='use gumbel softmax for message passing')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_act', action='store_true', help='use non-linearity for each layer')
    parser.add_argument('--use_jk', action='store_true', help='concat the layer-wise results in the final layer')
    parser.add_argument('--K', type=int, default=10, help='num of samples for gumbel softmax sampling')
    parser.add_argument('--tau', type=float, default=0.25, help='temperature for gumbel softmax')
    parser.add_argument('--lamda', type=float, default=0.1, help='weight for edge reg loss')
    parser.add_argument('--rb_order', type=int, default=0, help='order for relational bias, 0 for not use')
    parser.add_argument('--rb_trans', type=str, default='sigmoid', choices=['sigmoid', 'identity'],
                        help='non-linearity for relational bias')
    parser.add_argument('--batch_size', type=int, default=10000)