from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--load_dir', type=str, default='/data/chwang/Log/ordinal',
                            help='models are loaded from here')
        parser.add_argument('--epoch_count', type=int, default=60, help='which epoch checkpoint to load')
        self.isTrain = False
        return parser
