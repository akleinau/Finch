# FINCH
A web application to locally visualize higher-order feature interactions of black box models.

# Try out FINCH online
please visit https://huggingface.co/spaces/hztn/finch 

(you may have to restart the space if it is sleeping)


# Installation
- Install Python 3.11
- Clone the repository
- Run `pip install -r requirements.txt`
- Run `panel serve app.py --autoreload`


Currently, it supports loading all sklearn models saved with Python version 3.11 and sklearn version 1.5.1, and datasets saved as CSV. 