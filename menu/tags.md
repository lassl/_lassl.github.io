---
layout: page
title: tags
---

<div class="tags-frame">
    <div class="tags-list">
      {% for tag in site.tags %}
          <a href="#{{ tag[0] | slugify }}" class="tag-clouds">{{ tag[0] }}</a>
      {% endfor %}
    </div>
</div>

<ul class="posts">
  {% for tag in site.tags %}
    <h3 class="class-name" id="{{ tag[0] | slugify }}">
        <svg class="tags-item-icon" xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path><line x1="7" y1="7" x2="7.01" y2="7"></line></svg>
        {{ tag[0] }}
    </h3>
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
