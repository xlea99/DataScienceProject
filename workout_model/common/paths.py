import os
from pathlib import Path

# Helper method for testing the validity of a given pathToValidate in different ways.
def validatePath(pathToValidate: Path,  # The path to actually attempt to validate.
                 subPathsToTest: list = None,  # A list of subpaths to test for RELATIVE to the pathToValidate
                 readAccess: bool = True, writeAccess: bool = True, execAccess: bool = True, # Whether to test for certain accessibility
                 suppressErrors: bool = False,  # Whether to suppress errors, and return True/False instead.
                 createMissing = False # Whether to create a directory/any subdirectories if missing
                 ):
    # Test that the path actually exists
    if (not pathToValidate.exists()):
        if(createMissing):
            pathToValidate.mkdir(parents=True,exist_ok=True)
        elif(suppressErrors):
            return False
        else:
            raise ValueError(f"Path '{pathToValidate}' does not exist!")

    # Set up accessibility flag to be tested
    flags = 0
    if (not readAccess):
        flags |= os.R_OK
    if (not writeAccess):
        flags |= os.W_OK
    if (not execAccess):
        flags |= os.X_OK
    # Test accessibility of this path
    if (not os.access(str(pathToValidate), flags)):
        if (suppressErrors):
            return False
        else:
            raise PermissionError(f"Insufficient permissions for configured directory: {pathToValidate}")

    # Validate any subpaths provided
    if (subPathsToTest):
        for subPath in subPathsToTest:
            fullSubPath = pathToValidate / subPath
            if (not fullSubPath.exists()):
                if (suppressErrors):
                    return False
                else:
                    raise ValueError(f"File sub-structure for given path is invalid: {pathToValidate}")

    # Path is valid
    return True

# Simple class to validate and store program paths.
class Paths:

    # init generates all paths necessary for the object.
    def __init__(self):
        self.allPaths = {}
        self.add("appData",Path(os.environ['APPDATA']))

    # Method for getting the path with the given name.
    def get(self,pathname):
        return self.__getitem__(pathname)
    def __getitem__(self, key):
        lowerPathname = key.lower()
        if(lowerPathname in self.allPaths.keys()):
            return self.allPaths[lowerPathname]["Path"]
    # Method for registering a new path with the given name, path, and options.
    def add(self,pathname,path,subPathsToTest : list = None,suppressErrors = False,createMissing=False):
        if(pathname in self.allPaths.keys()):
            raise ValueError(f"Path name '{pathname}' already exists!")
        if(type(path) is not Path):
            path = Path(path)
        validatePath(pathToValidate=path,subPathsToTest=subPathsToTest,suppressErrors=suppressErrors,createMissing=createMissing)
        self.allPaths[pathname.lower()] = {"Path" : path}
    def __setitem__(self,key,value):
        self.add(pathname=key,path=value)
paths = Paths()

# Root path
thisFilePath = Path(__file__).resolve()
rootPath = thisFilePath.parent.parent.parent
paths["root"] = rootPath

# Other paths
paths["data"] = paths["root"] / "data"
paths["reports"] = paths["root"] / "reports"
