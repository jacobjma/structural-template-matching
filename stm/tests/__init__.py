import os
import sys
import unittest
from glob import glob
import tempfile

class ScriptTestCase(unittest.TestCase):
    def __init__(self, methodname='testfile', filename=None):
        unittest.TestCase.__init__(self, methodname)
        self.filename = filename

    def testfile(self):        
        with open(self.filename) as fd:
            exec(compile(fd.read(), self.filename, 'exec'), {})

    def id(self):
        return self.filename

    def __str__(self):
        return self.filename.split('test/')[-1]

    def __repr__(self):
        return "ScriptTestCase(filename='%s')" % self.filename

def get_tests(files=None):
    if files:
        files = [os.path.join(__path__[0], f) for f in files]
    else:
        files = glob(__path__[0] + '/*')
    
    tests = []
    for f in files:
        if f.endswith('.py'):
            tests.append(f)
    
    tests.sort()
    tests = [test for test in tests if not test.endswith('__.py')]
    
    return tests

def test(verbosity=1, testdir=None, stream=sys.stdout, files=None):
    """Main test-runner for STM."""
    
    tests = get_tests(files)
    
    ts = unittest.TestSuite()

    for test in tests:
        ts.addTest(ScriptTestCase(filename=os.path.abspath(test)))
    
    ttr = unittest.TextTestRunner(verbosity=verbosity, stream=stream)

    origcwd = os.getcwd()

    if testdir is None:
        testdir = tempfile.mkdtemp(prefix='stm-test-')
    else:
        if os.path.isdir(testdir):
            shutil.rmtree(testdir)
        os.mkdir(testdir)
    
    os.chdir(testdir)
    
    if verbosity:
        print('test-dir', testdir, '\n', file=sys.__stdout__)
    
    try:
        results = ttr.run(ts)
    finally:
        os.chdir(origcwd)
        sys.stdout = sys.__stdout__

    return results