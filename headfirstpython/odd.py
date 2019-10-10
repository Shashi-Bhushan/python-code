from datetime import datetime

import random
import time

if __name__ == '__main__':
    odds = range(1, 60, 2)

    for index in range(5):
        right_this_minute = datetime.today().minute

        if right_this_minute in odds:
            print(f"This minute {right_this_minute} seems a little odd.")
        else:
            print(f"This minute {right_this_minute} is not an odd minute.")

        wait_time = random.randint(1, 60)

        print(f"Waiting for {wait_time} seconds")
        time.sleep(wait_time)
