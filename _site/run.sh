#!/bin/bash

source $(brew --prefix)/opt/chruby/share/chruby/chruby.sh
chruby ruby-3.3.5
bundle exec jekyll serve
