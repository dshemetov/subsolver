"""Get Puzzles.

Get all puzzle text from Subsolver and write to 'puzzles.txt'.

It's not a simple HTTP GET because the puzzle text is rendered with
Javascript and requires a keypress to reveal the text. We got around this
with Selenium. This code uses Firefox, though you can probably modify it to
whatever browser you have.

If you're working remotely (via SSH), you will need to use the Selenium server.
To install:

1. Install Java:
    sudo apt install default-jre
2. Download Selenium server:
    wget https://github.com/SeleniumHQ/selenium/releases/download/selenium-4.15.0/selenium-server-4.15.0.jar
3. Start Selenium server:
    java -jar selenium-server-4.15.0.jar standalone
4. Set REMOTE_SSH = True below.
"""

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from webdriver_manager.firefox import GeckoDriverManager


def get_puzzle_text(puzzle_num: int, driver: webdriver.Firefox) -> str:
    driver.get(f"https://www.subsolver.com/classic/{puzzle_num}")

    # Press F and J to reveal the puzzle text
    ActionChains(driver).key_down("f").key_down("j").perform()
    ActionChains(driver).key_up("f").key_up("j").perform()

    s = "".join(e.text if e.text != "" else " " for e in driver.find_elements(By.CLASS_NAME, "puzzle-letter"))
    return s


if __name__ == "__main__":
    # Install the Gecko driver, if needed
    GeckoDriverManager().install()

    # Start the driver
    REMOTE_SSH = False
    if REMOTE_SSH:
        options = webdriver.FirefoxOptions()
        options.add_argument("--headless")
        driver = webdriver.Remote(options=options)
    else:
        driver = webdriver.Firefox()

    with open("puzzles.txt", "w") as f:
        for i in range(112):
            f.write(get_puzzle_text(i, driver) + "\n")
    driver.close()
