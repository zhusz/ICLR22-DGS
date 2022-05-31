import os


def returnExportedClasses(wishedClassNameList):  # To avoid import unnecessary class of different envs

    _testSuiteD = os.path.basename(os.path.dirname(__file__))

    exportedClasses = {}

    if wishedClassNameList is None or 'DTrainer' in wishedClassNameList:
        from ..trainer import Trainer
        class DTrainer(Trainer):
            pass
        exportedClasses['DTrainer'] = DTrainer

    return exportedClasses

