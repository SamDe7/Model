from selenium import webdriver
from selenium.webdriver.firefox.service import Service as ChromeService
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
import time  # Для добавления задержки

options = Options()
options.add_argument('--headless')
driver = webdriver.Firefox(service=ChromeService(), options=options)
driver.maximize_window()

url = 'https://unsplash.com/s/photos/Faces'
driver.get(url)

# Добавим небольшую задержку для загрузки изображений
time.sleep(5)

image_html_nodes = driver.find_elements(By.CSS_SELECTOR, '[data-testid="photo-grid-masonry-figure"]')
image_urls = []

for image_html_node in image_html_nodes:
    try:
        # Получаем URL из src
        image_url = image_html_node.get_attribute('src')
        # Получаем URL из srcset
        srcset = image_html_node.get_attribute('srcset')
        
        if srcset is not None:
            srcset_last_element = srcset.split(", ")[-1]
            image_url = srcset_last_element.split(" ")[0]
        
        if image_url:  # Проверяем, что URL не пустой
            image_urls.append(image_url)
    
    except StaleElementReferenceException as e:
        continue

print(image_urls)

driver.quit()

