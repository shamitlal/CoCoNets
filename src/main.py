# from model_carla_static import CARLA_STATIC
# from model_carla_flo import CARLA_FLO
# from model_carla_time import CARLA_TIME
# from model_carla_reloc import CARLA_RELOC
# from model_carla_sub import CARLA_SUB
# from model_carla_sob import CARLA_SOB
# from model_carla_explain import CARLA_EXPLAIN
import click #argparse is behaving weirdly
import os
import os
import cProfile
import logging

logger = logging.Logger('catch_all')

@click.command()
@click.argument("mode", required=True)
@click.option("--exp_name","--en", default="trainer_basic", help="execute expriment name defined in config")
@click.option("--run_name","--rn", default="1", help="run name")

def main(mode, exp_name, run_name):
    if mode:
        if "cm" == mode:
            mode = "CARLA_MOC"
        elif "cz" == mode:
            mode = "CARLA_ZOOM"
        else:
            raise Exception

    if run_name == "1":
        run_name = exp_name

    os.environ["MODE"] = mode
    os.environ["exp_name"] = exp_name
    os.environ["run_name"] = run_name


    import hyperparams as hyp
    from model_carla_moc import CARLA_MOC
    from model_carla_zoom import CARLA_ZOOM
    
    checkpoint_dir_ = os.path.join("checkpoints", hyp.name)


    if hyp.do_carla_moc:
        log_dir_ = os.path.join("logs_carla_moc", hyp.name)
    elif hyp.do_carla_zoom:
        log_dir_ = os.path.join("logs_carla_zoom", hyp.name)
    elif hyp.do_kitti_zoom:
        log_dir_ = os.path.join("logs_kitti_zoom", hyp.name)
    else:
        assert(False) # what mode is this?

    if not os.path.exists(checkpoint_dir_):
        os.makedirs(checkpoint_dir_)
    if not os.path.exists(log_dir_):
        os.makedirs(log_dir_)

    try:
        if hyp.do_carla_moc:
            model = CARLA_MOC(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_zoom:
            model = CARLA_ZOOM(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_kitti_zoom:
            model = KITTI_ZOOM(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        else:
            assert(False) # what mode is this?

    except (Exception, KeyboardInterrupt) as ex:
        logger.error(ex, exc_info=True)
        log_cleanup(log_dir_)

def log_cleanup(log_dir_):
    log_dirs = []
    for set_name in hyp.set_names:
        log_dirs.append(log_dir_ + '/' + set_name)

    for log_dir in log_dirs:
        for r, d, f in os.walk(log_dir):
            for file_dir in f:
                file_dir = os.path.join(log_dir, file_dir)
                file_size = os.stat(file_dir).st_size
                if file_size == 0:
                    os.remove(file_dir)

if __name__ == '__main__':
    main()
    # cProfile.run('main()')

