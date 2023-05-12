from unires.struct import settings

def get_sett(device, dir_results, pow, sr=True, label=None):
    s = settings()
    s.device = device
    s.dir_out = dir_results
    s.do_atlas_align = True
    s.fix = 0
    s.label = label  
    s.reg_scl = 1
    s.sched_num = 3
    s.scaling = False
    s.pow = pow
    if sr:
        s.prefix = 'sr_'
        s.max_iter = 512
    else:
        s.prefix = 'tri_'
        s.max_iter = 0        
    return s