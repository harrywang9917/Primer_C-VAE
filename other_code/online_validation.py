import numpy as np
import pandas as pd
import chromedriver_binary
from selenium import webdriver
import time
start = time.time()

forward_List = []
reverse_List = []
sbp_List = []

forward_add = './Sars-Cov-2 Project/v3_Dataset/result/Appearance_DataFrame.csv'
reverse_add = './Sars-Cov-2 Project/v3_Dataset/result/Appearance_DataFrame.csv'

forward = list(pd.read_csv(forward_add)['Unnamed: 0'])
reverse = list(pd.read_csv(reverse_add)['Unnamed: 0'])

browser = webdriver.Chrome()

for i in range(len(reverse)):
    for j in range(len(forward)):

        browser.get("https://genome.ucsc.edu/cgi-bin/hgPcr?wp_target=&db=0&org=SARS-CoV-2&wp_f=&wp_r=&wp_size=4000&wp_perfect=15&wp_good=15&wp_showPage=true&hgsid=1195519835_2f6DkETZqH6YjnorOmF9eB9vqJLF")
        time.sleep(1)

        Forward_Primer = browser.find_element_by_id('wp_f')
        Forward_Primer.click()
        Forward_Primer.send_keys(forward[j])

        Reverse_Primer = browser.find_element_by_id('wp_r')
        Reverse_Primer.click()
        Reverse_Primer.send_keys(reverse[i])

        submit = browser.find_element_by_id('Submit').click()
        time.sleep(1)

        result = browser.find_element_by_xpath(
            '//*[@id="firstSection"]/table/tbody/tr/td/table/tbody/tr/td/table/tbody/tr[2]/td[2]')
        result_word = result.text

        if 'No matches to' in result.text:
            pass
        else:
            forward_List.append(forward[j])
            reverse_List.append(reverse[i])
            sbp_List.append(result_word.split(' ')[1][:-2])

            forward_array = np.array(forward_List)
            reverse_array = np.array(reverse_List)
            sbp_array = np.array(sbp_List)

            successful_prime = np.vstack([forward_array, reverse_array])
            successful_prime = np.vstack([successful_prime, sbp_array]).T
            pd.DataFrame(successful_prime).to_csv('/Users/harry/Desktop/successful_prime_temp.csv', header=None, index=None)




elapsed = (time.time() - start)
hour = int(elapsed // 3600)
minute = int((elapsed % 3600) // 60)
second = int(elapsed - 3600 * hour - 60 * minute)
print("\n\n\nTime used:  {} hours {} minutes {} seconds  ---->  all  {} seconds".format(hour, minute, second, elapsed))
