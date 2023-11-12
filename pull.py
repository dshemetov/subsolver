from joblib import Memory
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from webdriver_manager.firefox import GeckoDriverManager

disk_cache = Memory(".subsolver")


@disk_cache.cache
def get_puzzle_text(puzzle_num: int, remote: bool = True) -> str:
    """Get puzzle text from Subsolver.

    Caches them to disk to be nice to the server.

    It's not a simple HTTP GET complicated because the puzzle text is rendered
    via Javascript and requires a keypress to reveal the text. If you're working
    remotely, requires you to install Firefox and a local Selenium server (which
    in turn requires Java):

        1. Ubuntu Java: sudo apt install default-jre
        2. Download Selenium server: https://www.selenium.dev/downloads/
        3. Probably need to install Firefox
        4. Start Selenium server: java -jar selenium-server-4.15.0.jar
           standalone

    """
    # Install the Gecko driver, if needed
    GeckoDriverManager().install()

    # Start the driver
    if remote:
        options = webdriver.FirefoxOptions()
        options.add_argument("--headless")
        driver = webdriver.Remote(options=options)
    else:
        driver = webdriver.Firefox()

    driver.get(f"https://www.subsolver.com/classic/{puzzle_num}")

    # Press F and J to reveal the puzzle text
    ActionChains(driver).key_down("f").key_down("j").perform()
    ActionChains(driver).key_up("f").key_up("j").perform()

    s = "".join(
        e.text if e.text != "" else " "
        for e in driver.find_elements(By.CLASS_NAME, "puzzle-letter")
    )

    driver.close()
    return s
