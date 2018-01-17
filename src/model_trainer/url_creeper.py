# encoding: utf-8
"""

UrlCreeper 0.0.1
@author: lacus x230-nz666
@usage: Using selenium for data crawling

"""
import sys
import json
import re
import time
import os
import pprint
import threading
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver import ActionChains
from selenium.common import exceptions as Sel_Exceptions
sys.path.append("../..")
from src import config

js_Actual_Height = """var body = document.body,
html = document.documentElement;

var height = Math.max(body.scrollHeight, body.offsetHeight, 
                       html.clientHeight, html.scrollHeight, html.offsetHeight);
return height;
"""
js_Scroll_Top = "return Math.max(document.body.scrollTop | document.documentElement.scrollTop);"
js_Window_Height = "return Math.max(document.body.clientHeight | document.documentElement.clientHeight);"

regex_url_pattern = "^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)([^#]*)?(#(.*))?"

class UrlCreeper(object):
    def __init__(self, url, browser=None, timeout=20):
        self.url = url
        self.browser = browser
        self.timeout = timeout
        self.wait = None

    def get_wait(self):
        if self.wait is None:
            self.wait = WebDriverWait(self.browser, self.timeout)
        return self.wait

    def crawl(self):
        self_create = False
        if self.browser is None:
            self.browser = get_browser()
            self_create = True
        self.get_wait()

        # dict used for collect result
        result = dict()

        # 首先尝试访问 url

        try:
            self.browser.get(self.url)
            self.wait.until(self.if_load_completed_helper)
        except Sel_Exceptions.TimeoutException:
            return None

        result["title"] = self.browser.title
        result["current_url"] = self.browser.current_url

        # 接下来根据一系列的规则重新处理每个页面，每个页面都是不一样的
        handle_functions = [
            self.handle_twitter,
            self.handle_instagram,
            self.handle_youtube_watch,
            self.handle_facebook,
            self.handle_telegraph_co,
            self.handle_independent_co,
            # you should always put the default
            # handler the last one.
            self.default_handler
        ]

        for handle_function in handle_functions:
            if handle_function(result["current_url"], result["title"], result):
                break

        if "404" in result or "timeout" in result or "archived" in result or "protected" in result:
            result = None

        print("{0}: {1}".format(self.url, result))
        if self_create:
            self.browser.close()
            self.browser = None
        return result

    def handle_twitter(self, new_url, title, result):
        if len(re.findall("twitter\.com", new_url)) > 0:
            if len(re.findall("Twitter / \?", title)) > 0:
                result["404"] = 1
            elif len(re.findall("^.+的 Twitter:.*$", title)) > 0:
                result["title"] = "adaptive media"
                result["is_media"] = 1

                # fetch all the review, if exists
                result["review"] = []
                try:
                    reply_tag = self.browser.find_element_by_css_selector("div.replies-to div.stream")
                    p_tags = reply_tag.find_elements_by_css_selector("p.tweet-text")

                    for p_tag in p_tags:
                        p = p_tag.text.strip()
                        if len(p) > 0:
                            result["review"].append(p)
                except Sel_Exceptions.NoSuchElementException:
                    pass

            elif len(re.findall("twitter\.com/safety/unsafe_link_warning", new_url)) > 0:
                result["title"] = "unsafe"
                result["archived"] = 1
            elif len(re.findall("protected_redirect=true", new_url)) > 0:
                result["protected"] = 1
            elif len(re.findall("account/suspended", new_url)) > 0:
                result["archived"] = 1
            else:
                print()
                print()
            return True
        else:
            return False

    def handle_instagram(self, new_url, title, result):
        if len(re.findall("instagram\.com", new_url)) > 0:
            self.wait.until(self.if_load_completed_helper)

            try:
                error_tag = self.browser.find_element_by_css_selector(".p-error,.dialog-404")
                result["404"] = 1
                return True
            except: pass

            result["review"] = []
            try:
                article_tag = self.browser.find_element_by_css_selector("article")
                review_panel_tag = article_tag.find_elements_by_xpath("div")[1]
                review_tags = review_panel_tag.find_elements_by_xpath("div")[0]
                for idx, review in enumerate(review_tags.find_elements_by_xpath("ul/li/span")):
                    p = review.text.strip()
                    if len(p) > 0:
                        result["review"].append(p)
                        if idx == 0:
                            result["title"] = p
            except: pass
            print("haha")
            return True
        else:
            return False

    def handle_youtube_watch(self, new_url, title, result):
        if len(re.findall("www\.youtube\.com/watch", new_url)) > 0:
            self.wait.until(self.if_load_completed_helper)
            time.sleep(2)
            try:
                error_tag = self.browser.find_element_by_css_selector("#error-screen")
                error_message = error_tag.text.strip()
                # try to feed some reviews
                result["review"] = []
                # scroll down the page
                self.auto_scrolling_down_helper()
                try:
                    for review_tag in self.browser.find_elements_by_css_selector("ytd-comments div#content"):
                        review = review_tag.text.strip()
                        if len(review) > 0:
                            result["review"].append(review)
                except: pass

                if len(error_message) > 0 and len(result["review"]) == 0:
                    result["404"] = 1
                    return True
            except: pass

            self.wait.until(expected_conditions.visibility_of_element_located((By.TAG_NAME, "h1")))
            result["title"] = self.browser.title.strip()
            self.wait.until(expected_conditions.visibility_of_element_located((By.CSS_SELECTOR, "div#content")))

            # get the video description
            content_tag = self.browser.find_element_by_css_selector("div#meta-contents div#content")
            result["description"] = content_tag.text.strip()

            # scroll down the page
            self.auto_scrolling_down_helper()

            # get all the review
            result["review"] = []
            for review_tag in self.browser.find_elements_by_css_selector("ytd-comments div#content"):
                review = review_tag.text.strip()
                if len(review) > 0:
                    result["review"].append(review)

            return True
        else:
            return False

    def handle_facebook(self, new_url, title, result):
        if len(re.findall("facebook\.com", new_url)) > 0:
            try:
                self.wait.until(self.if_load_completed_helper)
            except Sel_Exceptions.TimeoutException:
                result["timeout"] = 1
                return True

            if len(re.findall("facebook\.com/photo.php", new_url)) == 0 and \
                            len(re.findall("facebook\.com/.+/posts", new_url)) == 0:
                result["protected"] = 1
                return True

            try:
                user_content_tag = self.browser.find_element_by_css_selector("div.userContentWrapper")
                post_tag = user_content_tag.find_element_by_xpath("div[1]/div[2]/div[2]")
                result["body"] = post_tag.text.strip()

                time.sleep(1.5)

                close_button = self.get_css_item_helper("a#expanding_cta_close_button")
                try:
                    self.scroll_and_click_helper(close_button)
                except:
                    print("failed")
                    result["404"] = 1

                try:
                    review_button = user_content_tag.find_element_by_xpath("div[2]")
                    a_tag = review_button.find_element_by_tag_name("a")
                    self.scroll_and_click_helper(a_tag)
                    result["review"] = []

                    # if succeeded, then some of the reviews display.

                    # find the view more button, if succeeded
                    try:
                        view_more_a = self.get_css_item_helper("a.UFIPagerLink", user_content_tag)
                        self.scroll_and_click_helper(view_more_a)
                        time.sleep(3)
                    except:
                        pass

                    # find all the expand button, if succeeded
                    try:
                        for a in user_content_tag.find_elements_by_css_selector("a.fss"):
                            self.scroll_and_click_helper(a)
                        time.sleep(1)
                    except: pass

                    review_tags = self.get_css_items_helper("span.UFICommentBody", user_content_tag)
                    for review_tag in review_tags:
                        review = review_tag.text.strip()
                        if len(review) > 0:
                            result["review"].append(review)
                except: pass

            except Sel_Exceptions.NoSuchElementException:
                result["404"] = 1

            return True
        else:
            return False

    def handle_independent_co(self, new_url, title, result):
        if len(re.findall("independent\.co\.uk", new_url)) > 0:
            try:
                header_tag = self.get_css_item_helper("article header")
                h1_tag = header_tag.find_element_by_css_selector("h1")
                intro_p_tag = header_tag.find_element_by_css_selector("div.intro p")
                result["title"] = h1_tag.text.strip()
                main_content_tag = self.get_css_item_helper("div.main-content-column")
                result["subtitle"] = h1_tag.text.strip()
                result["body"] = []
                for p_tag in main_content_tag.find_elements_by_css_selector("p"):
                    p = p_tag.text.strip()
                    if len(p) > 0:
                        result["body"].append(p)
            except Sel_Exceptions.TimeoutException:
                result["timeout"] = 1
            return True
        else:
            return False

    def handle_telegraph_co(self, new_url, title, result):
        if len(re.findall("telegraph\.co\.uk/news", new_url)) > 0:
            try:
                h1_tag = self.get_css_item_helper("div.storyHead h1")
                h2_tag = self.get_css_item_helper("div.storyHead h2")
                result["title"] = h1_tag.text.strip()
                result["subtitle"] = h2_tag.text.strip()
                mainBodyArea_tag = self.get_css_item_helper("div#mainBodyArea")
                para_list = mainBodyArea_tag.find_elements_by_css_selector("p")
                result["body"] = []
                for p_tag in para_list:
                    p = p_tag.text.strip()
                    if len(p) > 0:
                        result["body"].append(p)
            except Sel_Exceptions.TimeoutException:
                result["timeout"] = 1
            return True
        else:
            return False

    def default_handler(self, new_url, title, result):
        try:
            self.wait.until(self.if_load_completed_helper)
            result["default_handler"] = 1
            result["title"] = self.browser.title
        except Sel_Exceptions.TimeoutException:
            result["timeout"] = 1
        return True

    # helpers
    def get_css_item_helper(self, locator, parent=None):
        self.wait.until(expected_conditions.visibility_of_element_located((By.CSS_SELECTOR, locator)))
        if parent is None:
            parent = self.browser
        return parent.find_element_by_css_selector(locator)

    def get_css_items_helper(self, locator, parent=None):
        self.wait.until(expected_conditions.visibility_of_element_located((By.CSS_SELECTOR, locator)))
        if parent is None:
            parent = self.browser
        return parent.find_elements_by_css_selector(locator)

    def auto_scrolling_down_helper(self, step=40):
        complete_count = 0
        while complete_count < 9:
            scr_top = self.browser.execute_script(js_Scroll_Top)
            win_height = self.browser.execute_script(js_Window_Height)
            act_height = self.browser.execute_script(js_Actual_Height)
            avail_top = act_height - win_height
            if avail_top > scr_top:
                for top in range(min(scr_top + step, avail_top), avail_top, step):
                    scr_top = self.browser.execute_script(js_Scroll_Top)
                    if scr_top >= top:
                        continue
                    self.browser.execute_script("window.scrollTo(0,%d)" % top)
                    time.sleep(0.2)
                self.browser.execute_script("window.scrollTo(0,%d)" % avail_top)
                complete_count = 0
            else:
                complete_count += 1
            time.sleep(1)

    def scroll_and_click_helper(self, item):
        self.browser.execute_script("window.scrollTo(%d,%d)" % (0, item.location["y"]))
        time.sleep(0.5)
        item.click()

    @staticmethod
    def if_load_completed_helper(driver):
        return driver.execute_script("return document.readyState") == "complete"


def get_browser():
    return webdriver.Chrome()


def main(urls):
    browser = get_browser()

    url_cache = None
    try:
        url_cache = json.load(open(config.URL_CACHE_PATH, "r"))
    except FileNotFoundError:
        url_cache = dict()

    for url in urls:
        if url in url_cache.keys():
            continue
        creeper = UrlCreeper(url, browser, 60)
        ret = creeper.crawl()
        if ret is not None:
            url_cache[url] = ret
            json.dump(url_cache, open(config.URL_CACHE_PATH, "w"), indent=4)

    browser.close()


def fetch_all_urls_map():
    """
    For all the compressed urls into its real urls.

    Browser may time out unexpectedly, so you must run this program
    :return: None
    """
    file_path = os.path.join(config.CWD, "url_set")
    url_dict = dict()
    all_url_set = set()
    try:
        url_dict = json.load(open(file_path, "r"))
        for d, url_list in url_dict.items():
            if d == "t.co":
                continue
            for url in url_list:
                all_url_set.add(url.strip())
        if "t.co" in url_dict:
            del url_dict["t.co"]
    except: pass
    chrome = get_browser()
    tweets = json.load(open(config.PROCESSED_TRAIN, "r"))
    rc = re.compile(regex_url_pattern)
    for idx, tweet in enumerate(tweets):
        url_list = tweet["twitter_url"]
        for url in url_list:
            url = url.strip()
            if url in all_url_set:
                continue
            new_url = url
            try:
                chrome.get(url)
                new_url = chrome.current_url
            except Sel_Exceptions.TimeoutException:
                pass
            all_url_set.add(url)
            domain = rc.findall(new_url)[0][3]
            print("{0} --> {1}".format(url, domain))
            url_dict.setdefault(domain, [])
            url_dict[domain].append(url)
            json.dump(url_dict, open(file_path, "w"), indent=4)
    pprint.pprint(url_dict)
    chrome.close()


if __name__ == "__main__":
    file_path = os.path.join(config.CWD, "url_set")
    url_dict = json.load(open(file_path, "r"), encoding="utf-8")
    url_list = list(url_dict.items())
    url_list.sort(key=lambda x: -len(x[1]))

    # creeper = UrlCreeper("http://t.co/EJaFcMsIrA")
    # creeper.crawl()

    # value = url_list[3]
    # urls = value[1]
    # br = get_browser()
    # for i, _url in enumerate(urls):
    #     creeper = UrlCreeper(_url, br)
    #     creeper.crawl()

    # for youtube
    # value = url_list[3]
    # urls = value[1]
    # main(urls)

    for domain_url, urls in url_list:
        print(domain_url)
        print("==" * 20)
        main(urls)
