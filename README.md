# ddddocr_web

## 本仓库内容

### 移植到网页中的 ddddocr

详见 demo.html，在浏览器中直接使用 ONNX Runtime Web 运行 ddddocr 模型进行验证码识别，无需服务器

### ddddocr UInt8 量化模型

见 common_q8.onnx 文件，效果对比原模型略有下降，不过对大部分验证码识别率依旧较高。体积小，对 Web 友好。

### 借助上述模型实现的自动验证码填充油猴脚本（无需服务器，纯本地计算）

见 captcha.user.js，可以自动帮忙填写验证码

脚本能自动查找所有网站上的验证码并尝试填写，也可关闭自动查找手动添加规则，以下以 NJU 统一认证为例

<img src="https://github.com/lyc8503/ddddocr_web/raw/refs/heads/master/demo.webp" width="600">

### 致谢

感谢 @sml2h3 制作的 [ddddocr](https://github.com/sml2h3/ddddocr) 项目作为本项目优质模型来源  
感谢 @Do1e 的 [NJUcaptcha](https://github.com/Do1e/NJUcaptcha) 项目为本项目提供灵感来源与部分代码参考  
感谢 @crab 编写的[万能验证码自动输入（升级版）](https://greasyfork.org/zh-CN/scripts/418942-%E4%B8%87%E8%83%BD%E9%AA%8C%E8%AF%81%E7%A0%81%E8%87%AA%E5%8A%A8%E8%BE%93%E5%85%A5-%E5%8D%87%E7%BA%A7%E7%89%88)脚本，本项目的脚本以此修改而来
