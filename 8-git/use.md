## 生成公钥和私钥
- ssh-keygen -t ed25519 -C "你的GitHub邮箱" -f ~/.ssh/id_ed25519_github
- ssh-keygen -t ed25519 -C "你的公司邮箱" -f ~/.ssh/id_ed25519_gitlab_company
- 将公钥(pub)上传


## 常用命令
git reset --hard origin/release_5.61.x   强制本地和远程一样
git branch -u origin main:lgc/main       将本地的main推送到远程的lgc/main
git config user.email "610826334@qq.com"
git reset --soft HEAD~1



feat      新功能 feature
fix       修复 bug
docs      文档修改 documentation
style     代码格式修改，不影响逻辑
refactor  重构，不新增功能也不修 bug
perf      性能优化 performance
test      测试相关

## 问题:
- git reset 的用法是什么？

- git merge和git rebase 有什么区别？