---
layout: page
title: authors
---

<div class="tags-frame">
  {% for author in site.authors %}
      <a href="#{{ author[0] | slugify }}" class="tag-clouds">{{ author[0] }}</a>
  {% endfor %}
</div>

<ul class="posts">
  {% for author in site.authors %}
    <h3 class="class-name" id="{{ author[0] | slugify }}">{{ author[0] }}</h3>
    {% for post in site.posts %}
        {% if post.author == author[0] %}
            <li itemscope>
              <a class="title-name-in-list" href="{{ site.github.url }}{{ post.url }}">{{ post.title }}</a>
              <p class="post-date"><span>written by {{ post.author }}
              <i class="fa fa-calendar" aria-hidden="true"></i> {{ post.date | date: "%Y %B %-d" }} - <i class="fa fa-clock-o" aria-hidden="true"></i> {% include read-time.html %}</span></p>
            </li>
        {% endif %}
    {% endfor %}
  {% endfor %}
</ul>
