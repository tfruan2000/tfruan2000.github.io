# tfruan's Blog

> Powered by [Chirpy Jekyll Theme](https://github.com/cotes2020/jekyll-theme-chirpy)
>
> Deployed with [Github Pages](https://pages.github.com/)

# Blog Site

Please visit [tfruan2000.github.io](https://tfruan2000.github.io/)

# deploy on local

```bash
bundle install

# construct
bundle exec jekyll server

# check
bundle exec htmlproofer _site --disable-external --ignore-urls "/^http:\/\/127.0.0.1/,/^http:\/\/0.0.0.0/,/^http:\/\/localhost/"
```