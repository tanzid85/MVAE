from scipy.io import loadmat

class MotionVAEOption(object):
   
    # Dataset
    dataset_dir = '/home/mdtanzid/PycharmProjects/MVAE/Data'
    database_ratio = 1.0
    pose_feature = ['joint_pos', 'joint_velo']
    update_joint_pos = False
    predict_phase = False
    
    # Network
    frame_size = None
    latent_size = 4  # Need to talk to Dr. Zhang about it
    hidden_size = 256
    num_condition_frames = 1

    num_future_predictions = 1
    num_experts = 6

    # Train
    gpu_ids = [0]
    base_opt_ver = None
    model_base_ver = None
    nframes_seq = 10

    nseqs = 200
    curriculum_schedule = None
    mixed_phase_schedule = None
    weights = {'recon': 1, 'kl': 1, 'recon_phase': 10}
    softmax_future = False

    batch_size = 64
    num_threads = 8
    n_epochs = 1000
    n_epochs_decay = 200

    log_freq = 50
    vis_freq = 1e9
    save_freq_epoch = 100
    lr = 0.0001

    # checkpoint_dir = 'results/motionVAE'
    continue_train = False
    use_amp = False
    no_log = False

    # Test
    test_only = False
    result_dir = '/home/mdtanzid/PycharmProjects/MVAE/results'
    # infer_racket = False


    def __init__(self):
        # Add all class attributes as instance attributes
        for key in sorted(dir(self)):
            if not key.startswith('__'):
                setattr(self, key, getattr(self, key))


    def update(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])


    def print(self):
        for key in sorted(self.__dict__):
            if not key.startswith('_'):
                print("Option: {:30s} = {}".format(key, self.__dict__[key]))


    def load(self, version):
        stack = [motion_vae_opt_dict[version]]
        while 'base_opt_ver' in stack[-1]:
            stack.append(motion_vae_opt_dict[stack[-1]['base_opt_ver']])
        stack = stack[::-1]
        for opt_update in stack:
            self.update(**opt_update)


motion_vae_opt_dict = {
'allegro_hand': {
    'model_ver'                             : 'allegro_hand',
    'pose_feature'                          : ['joint_pos', 'joint_velo'],
    'update_joint_pos'                      : False,
    'predict_phase'                         : True,
    'frame_size'                            : 32,
    'num_condition_frames'                  : 1,
    'num_future_predictions'                : 1,
    'nframes_seq'                           : 10,
    'batch_size'                            : 10,
    'nseqs'                                 : 200,
    'softmax_future'                        : True,
    'curriculum_schedule'                   : [0.1, 0.2],
    'mixed_phase_schedule'                  : [(0, 1), (0.5, 0.1)],
    'weights'                               : {'recon': 1, 'kl': 0.5, 'recon_phase': 10},
    'n_epochs'                              : 1000,
    'n_epochs_decay'                        : 200,
    'save_freq_epoch'                       : 50,
},
}
opt = MotionVAEOption()