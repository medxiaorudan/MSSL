import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[0]


class MyPath(object):
    """
    User-specific path configuration.
    """
    @staticmethod
    def db_root_dir(database=''):
        db_root = '/data/morpheme/user/rxiao/MIDL/bigdata/'
        db_names = {'RCC'}

        if database in db_names:
            return os.path.join(db_root, database)
        
        elif not database:
            return db_root
        
        else:
            raise NotImplementedError

    @staticmethod
    def seism_root():
        return '/data/morpheme/user/rxiao/MIDL/deep_learning/Multi-task/evaluation/seism/'
