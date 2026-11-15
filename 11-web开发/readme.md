# 1、脚手架搭建
- 1)安装Node.js:     https://nodejs.org/zh-cn/download  自行下载安装包进行安装
    验证是否安装成功： node -v

- 2)包管理工具npm:  类似pip  安装Node以后默认集成了npm
    验证是否安装成功: npm -v


# 创建项目:https://cn.vuejs.org/guide/quick-start.html
1、npm create vue@latest
2、cd vue-project
   npm install
   npm run dev


# 项目文件解读
node_modules          项目依赖的文件夹
public                公共文件，一般不动
src                   源码目录
src/components        公共组件
main.ts               项目的入口文件
App.vue               单文件组件
index.html            唯一的页面,单页面应用
package-lock.json     锁定依赖包的版本号
package.json          项目的版本号
vite.config.ts        vite的配置文件


# 设置下载源
npm config set registry https://registstry https://registry.npmirror.com

# vscode插件
vue-offical
vscode-icons
live server
vue vscode snippets


# 调试界面
F12
