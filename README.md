# CosmoCHE.github.io
My Personal Blog

Fork from [jamesnulliu.github.io](https://github.com/jamesnulliu/jamesnulliu.github.io)

## Create Blog Post
To create a new blog post, you need to have an archetype file located at `archetypes/blog.md`. Here is an example of what the content of that file should look like:
```
---
title: '{{ replace .File.ContentBaseName "-" " " | title }}'
date: {{ .Date }}
draft: false
author: ["Kosmo CHE"]
keywords: 
    - xxx
categories:
    - xxx
tags:
    - xxx
description: "This blog post discusses xxx"
summary: "This blog post discusses xxx"
comments: true
images:
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
```
To create a new blog post, use the following command in the terminal:
```
hugo new --kind blog content/blogs/{blogs-title}/index.{language}.md
```
language can be `en` or `zh`, and the `{blogs-title}` should be in lowercase and hyphenated (e.g., `a-brief-talk-on-dpo`).
