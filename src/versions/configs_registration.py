def getConfigGlobal(P, D, S, R, wishedClassNameList=None):
    returnExportedClasses = __import__(
        '%s.%s.extension' % (P, D), globals(), locals(), ['returnedExportedClasses']
    ).returnExportedClasses
    exportedClasses = returnExportedClasses(wishedClassNameList)
    if S == 'SNono':
        getConfigFunc = lambda P, D, S, R: None
    else:
        getConfigFunc = __import__(
            '%s.%s.config_%s' % (P, D, S), globals(), locals(), ['getConfigFunc']
        ).getConfigFunc

    return {
        'getConfigFunc': getConfigFunc,
        'exportedClasses': exportedClasses,
    }
