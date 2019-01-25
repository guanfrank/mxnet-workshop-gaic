# QWIKLab Instructions

**<<<< This workshop is now complete and QWIKLab setup is not active anymore!! >>>>>**

Welcome to Apache MXNet Bootcamp. We will use Amazon SageMaker to run the Apache MXNet lab sessions.

1.	Go to [https://events-aws.qwiklabs.com/](https://events-aws.qwiklabs.com/) and create an account.
2.	Send the email-id, with which you registered in qwiklabs, to
**wamy@amazon.com** and **kannanva@amazon.com** (Your QWIKLab account will be added to the student list in the lab. This will essential to get access to the lab.)
3.	Once you login to QwikLabs, click on the MXNet/Gluon BootCamp Lab
    ![Qwiklabs Gluon Lab](./lab/assets/qwiklabs_first_page.png)

4. Once inside the lab, use the Notebook Instance and the credentials (Account ID, User Name, Password) provided to login to Amazon SageMaker jupyter notebook.
    ![Qwiklabs Sage Maker](./lab/assets/qwik_notebook.png)

5. We will download the lab material to use on the notebook, start a new terminal session on SageMaker.
    ![Qwiklabs Sage Maker Terminal](./assets/qwik_lab.png)

6. Open a terminal from the Launcher
    ![Qwiklabs Terminal](./assets/qwik_terminal.png)

7. Clone the github repo recursively – it contains an assortment of material used in this workshop.
    ```
    cd ~/SageMaker
    git clone --recursive https://github.com/vandanavk/mxnet-workshop-gaic.git
    ```

8. 8.	To access the exercises, go to mxnet-workshop-gaic/lab directory. Choose conda_mxnet_p36 environment as the Kernel when executing a notebook. 
![Qwiklabs Sage Maker Conda Env](./assets/qwik_kernel.png)

You are all set! Let's get started...
