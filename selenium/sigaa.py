from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time

def main():
    driver = webdriver.Chrome()
    driver.get("https://autenticacao.ufrn.br/sso-server/login?service=https%3A%2F%2Fsigaa.ufrn.br%2Fsigaa%2Flogin%2Fcas")
    time.sleep(3)

    username = driver.find_element(By.ID, "username")
    password = driver.find_element(By.ID, "password")

    username.send_keys("") #login do sigaa
    password.send_keys("")    #senha do sigaa

    #driver.find_element(By.NAME, "Entrar").click()
    password.send_keys(Keys.RETURN)

    time.sleep(5)
    driver.quit()

if __name__ == "__main__":
    main()