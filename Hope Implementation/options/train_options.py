from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_freq', type=int, default=1,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=20,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--epoch_count', type=int, default=0, help='epoch count')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_decay', type=float, default=0.95, help='initial lambda decay value')
        parser.add_argument('--lr_policy', type=str, default='exp',
                            help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--interpolation_lambda', type=float, default=20.0, help='interpolation strength')
        parser.add_argument('--sigreg_weight', type=float, default=0.09, help='SigREG loss coefficient weight')
        parser.add_argument('--sigreg_knots', type=int, default=17, help='number of knots for SigREG')
        parser.add_argument('--sigreg_num_proj', type=int, default=1024, help='number of projections for SigREG')
        self.isTrain = True
        return parser
