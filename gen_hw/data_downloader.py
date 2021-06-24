import requests
import re
import os
import time
import threading
from bs4 import BeautifulSoup
import json
import argparse
from tqdm import tqdm

root_path = root_path = './data/'
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--type", type=str, default="0")
parser.add_argument("--name", type=str, default=None)
args = parser.parse_args()

"""
USAGE:
python data_downloader.py --name file_name --type data/model
"""

# 防止网络中断的增强型requests
def request(url, headers, stream=False, trycnt=5, sleep_t=2):
    flag = True
    r_cnt = 1
    while trycnt >= 0:
        try:
            t_r = requests.get(url, headers=headers, stream=stream)
            tmpl = int(t_r.headers['content-length'])
            t_r.raise_for_status()
            trycnt = -1
        except:
            if trycnt <= 0:
                if flag:
                    trycnt = 5
                    flag = False
                    time.sleep(120)
                else:
                    print('Break down!')
                    #raise RuntimeError
                    time.sleep(3600)
                    trycnt = 5
            else:
                if flag:
                    trycnt -= 1
                    time.sleep(0.5)
                else:
                    trycnt -= 1
                    time.sleep(sleep_t + r_cnt)
                    r_cnt *= 2
    return t_r

# 线程句柄
def Handler(start, end, url, filename, headers={}):
    tt_name = threading.current_thread().getName()
    # print(tt_name + ' 开始下载')
    r = request(url, headers, True, sleep_t=2)
    with open(filename, 'r+b') as fp:
        h_try_cnt = 5
        h_cnt = 1
        while h_try_cnt >= 0:
            try:
                fp.seek(start)
                for chunk in r.iter_content(102400):
                    if chunk:
                        fp.write(chunk)
                    else:
                        raise ValueError
                h_try_cnt = -1
            except:
                if h_try_cnt <= 0:
                    print("Network Error")
                    # raise ValueError
                    time.sleep(3600)
                    h_try_cnt = 5
                else:
                    h_try_cnt -= 1
                    time.sleep(2 + h_cnt)
                    h_cnt *= 2
    
# 对单个文件进行多线程
def scrape(name, type):
    headers = {
        'accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'user-agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36'
    }
    vpath = root_path
    if not os.path.exists(vpath):
        os.makedirs(vpath)
    # 下载
    file_name = name
    npath = vpath + file_name
    if type == 'model':
        cur_url = 'https://datasets.flagwyx.workers.dev/pre-trained%20models/'
    elif type == 'data':
        cur_url = 'https://datasets.flagwyx.workers.dev/datasets/'
    cur_url += name
    r_test = request(cur_url, headers, True)
    all_thread = 1 # 至少有一个线程
    file_size = int(r_test.headers['content-length'])
    # 输出文件大小
    if file_size:
        print('文件大小：' + str(int(file_size / 1024 / 1024)) + "MB")
    else:
        print('未获得文件')
        return
    starttime = time.time()
    # 已经下载完毕，就不用下载了
    if (not os.path.exists(npath)) or os.path.getsize(npath) < file_size:
        fp = open(npath, 'wb')
        #fp.truncate(file_size)
        fp.close()
        # 每个线程每次下载大小为10M
        size = 104857600
        # 单个文件大于10M, 使用多线程
        if file_size > size:
            all_thread = int(file_size / size)
            # 线程数为15
            if all_thread > 10:
                all_thread = 10
        # 每个线程下载文件的大小
        part = file_size // all_thread
        threads = []
        for i in range(all_thread):
            # 获取每个线程开始时的文件位置
            start = part * i
            # 获取每个文件结束位置
            if i == all_thread - 1:
                end = file_size
            else:
                end = start + part
            if i > 0:
                start += 1
            headers = headers.copy()
            headers['Range'] = "bytes=%s-%s" % (start, end)
            t = threading.Thread(target=Handler, name='th-' + str(i),
                                    kwargs={'start': start, 'end': end, 'url':cur_url, 'filename': npath, 'headers': headers})
            t.setDaemon(True)
            threads.append(t)
        # 线程开始
        for t in threads:
            time.sleep(0.1)
            t.start()

        # 等待所有线程结束
        for t in tqdm(threads):
            t.join()
    endtime = time.time()
    if os.path.getsize(npath) >= file_size:
        print('下载完成！用时：%s秒' % (endtime - starttime))
    else:
        print('下载失败')

if __name__ == '__main__':
    scrape(args.name, args.type)
