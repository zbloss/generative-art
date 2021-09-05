from selenium.webdriver import Firefox, Chrome
from selenium.webdriver.chrome.options import Options
from selenium import webdriver

from glob import glob
import os
import time
from tqdm.notebook import tqdm

SELENIUM_SESSION_FILE = './selenium_session'
SELENIUM_PORT=9515

def build_driver():
    options = Options()
    options.add_argument("--disable-infobars")
    options.add_argument("--enable-file-cookies")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_extension('C:/Users/altoz/Downloads/coinbase_wallet.crx')

    if os.path.isfile(SELENIUM_SESSION_FILE):
        session_file = open(SELENIUM_SESSION_FILE)
        session_info = session_file.readlines()
        session_file.close()

        executor_url = session_info[0].strip()
        session_id = session_info[1].strip()

        capabilities = options.to_capabilities()
        driver = webdriver.Remote(command_executor=executor_url, desired_capabilities=capabilities)
        # prevent annoying empty chrome windows
        driver.close()
        driver.quit() 

        # attach to existing session
        driver.session_id = session_id
        return driver

    driver = webdriver.Chrome(options=options, port=SELENIUM_PORT)

    session_file = open(SELENIUM_SESSION_FILE, 'w')
    session_file.writelines([
        driver.command_executor._url,
        "\n",
        driver.session_id,
        "\n",
    ])
    session_file.close()

    return driver

browser = build_driver()
browser.execute_script('''window.open("https://opensea.io/","_blank");''')

x = input("Please log in to your ETH wallet, then click [ENTER] to continue")
while True:
    try:
        browser.get('https://opensea.io/assets/monsters-evolution?collectionSlug=monsters-evolution&search[sortAscending]=false&search[sortBy]=LISTING_DATE')
        
        for _ in range(100):
            browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            time.sleep(1)

        browser.find_element_by_class_name('AssetMedia--img').click()
        time.sleep(4)

        buttons = browser.find_elements_by_xpath("//*[contains(text(), 'Lower price')]")
        time.sleep(4)

        buttons = browser.find_elements_by_xpath("//*[contains(text(), 'Sell')]")
        buttons[0].click()
        time.sleep(4)
        
        browser.find_element_by_class_name('Input--input').send_keys('0.1')
        buttons = browser.find_elements_by_xpath("//*[contains(text(), 'Post your listing')]")
        buttons[0].click()
        time.sleep(8)
        
    except:
        #time.sleep(10)
        pass