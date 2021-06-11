from datetime import datetime
import os


def current():
    # datetime object containing current date and time
    now = datetime.now()    # dd/mm/YY_H:M:S
    date_time = now.strftime("%d/%m/%Y_%H.%M.%S")
    return date_time


def checkFileExist(filename, create=False):
    path = os.getcwd()
    if not os.path.exists(filename):
        if create:
            os.makedirs(filename)
            return os.path.join(path, filename)
        else:
            return None

    else:
        return os.path.join(path, filename)
