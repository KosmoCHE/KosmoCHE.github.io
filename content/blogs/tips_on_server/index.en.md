---
title: 'Tips on Server(The suck jupyter kernel)'
date: 2025-11-11T19:29:20+08:00
draft: false
author: ["Kosmo CHE"]
categories:
    - Tools & Tips​
tags:
    - Server Tips
    - Jupyter
description: "This blog is to note some tips and command lines for server management and Jupyter usage."
comments: true
images:
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
## Introduction
Some commands and tips, often used when working on a server, I feel tired of asking gpt again and again.

### 1. Kill Jupyter Notebook Kernel
When you use Jupyter Notebook on a server,especially when in VSCode, sometimes the kernel may get stuck or become unresponsive.
Unfortunately, VSCode does not provide a direct way to kill or shutdown the kernel from its interface.This make a lot of jupyter processes running on the server, consuming resources and causing confusion.

Check the running Jupyter processes with:
```bash
pgrep -a -u username -f jupyter
```
and then kill 
```bash
kill -9 PID
```
And if you want to be more aggressive, the most brutal method is:
```bash
pkill -9 -u username -f jupyter
```



