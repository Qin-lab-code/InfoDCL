
import time as time
from DataHandler import DataHandler
from Coach import Coach

if __name__ == "__main__":

    handler = DataHandler()
    coach = Coach(handler)
    coach.run()