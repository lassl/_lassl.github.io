---
layout: page
title: categories
---

<div class="tags-frame">
  {% for tag in site.categories %}
      <a href="#{{ tag[0] | slugify }}" class="tag-clouds">{{ tag[0] }}</a>
  {% endfor %}
</div>

<ul class="posts">
  {% for tag in site.categories %}
    <h3 class="class-name" id="{{ tag[0] | slugify }}">{{ tag[0] }}</h3>
    {% for post in tag[1] %}
        <li itemscope>
          <a class="title-name-in-list" href="{{ site.github.url }}{{ post.url }}">{{ post.title }}</a>
          <p class="post-date"><span>written by
          {% if post.author %}
            {{ post.author }}
          {% else %}
            {{ site.author }}
          {% endif %}
          <i class="fa fa-calendar" aria-hidden="true"></i> {{ post.date | date: "%Y %B %-d" }} - <i class="fa fa-clock-o" aria-hidden="true"></i> {% include read-time.html %}</span></p>
        </li>
    {% endfor %}
  {% endfor %}
</ul>
