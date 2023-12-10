from motion_vae.config import *
from motion_vae.base import MotionVAEModel 
from motion_vae.test import test_motion_vae_randomwalk
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('-e', '--exp', action='store', dest='exp_list',
                    type=str, nargs='*', default=['vx.x.x'],
                    help="Examples: -e v8.3.1.0")
parser.add_argument('-p', '--pre-run', action='store_true')
parser.add_argument('--no_log', action='store_true')
parser.add_argument('-t', '--run_test', action='store_true')

parser.add_argument('-to', '--test_only', action='store_true')
parser.add_argument('-i', '--interactive', action='store_true')
parser.add_argument('-n', '--nframes', action='store', type=int, default=1000)
parser.add_argument('-a', '--nactors', action='store', type=int, default=5)
parser.add_argument('--ntests', action='store', type=int, default=1)
parser.add_argument('--suffix', action='store', type=str, default='')

args = parser.parse_args()


if __name__ == '__main__':

    for version in args.exp_list:
        # load option 
        opt = MotionVAEOption()
        opt.load('allegro_hand')
        if args.pre_run:
            opt.quick_load = True
            opt.n_epochs = 1000
            opt.n_epochs_decay = 200
            opt.nseqs = 200
            opt.no_log = True
        opt.no_log = args.no_log

        if not args.test_only:
            motion_vae = MotionVAEModel(opt)

            # Print the device information
            print(f"Model is running on device: {motion_vae.device}")
            print(motion_vae.model)
            opt.print()
            print("The training is starting soon...")
            motion_vae.train()
            print("The training is finished. Well done!")
            x = 1
        # run test
        if args.run_test or args.test_only:
            print("Running test ... ")
            opt.test_only = True
            opt.infer_racket = True
            test_motion_vae_randomwalk(opt, 
                same_init_state=True, 
                num_test=args.ntests, 
                num_runner=args.nactors, 
                result_dir_suffix=args.suffix,
                nframes=args.nframes, 
                interactive=args.interactive
            )