// ==UserScript==
// @name         验证码自动输入
// @namespace    https://github.com/lyc8503/ddddocr_web
// @version      5.0.20250822
// @description  自动查找、识别、填写验证码的小脚本，使用 ONNX 模型本地识别，无需连接后端服务器。安装时需要从 GitHub 拉取模型，请确保网络畅通。如遇较多自动查找错误影响使用，可在设置中关闭自动查找，对常用网站手动添加规则。
// @author       crab & lyc8503
// @match        http://*/*
// @match        https://*/*
// @require      http://libs.baidu.com/jquery/2.0.0/jquery.min.js
// @require      http://ajax.aspnetcdn.com/ajax/jquery/jquery-2.0.0.min.js
// @resource     cktools https://greasyfork.org/scripts/429720-cktools/code/CKTools.js?version=1034581
// @require      https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js
// @resource     CAPTCHA_MODEL https://raw.githubusercontent.com/lyc8503/ddddocr_web/refs/heads/master/common_q8.onnx
// @resource     ort-wasm-simd-threaded.jsep.mjs https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort-wasm-simd-threaded.jsep.mjs
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        GM_listValues
// @grant        GM_openInTab
// @grant        GM_registerMenuCommand
// @grant        GM_unregisterMenuCommand
// @grant        GM_xmlhttpRequest
// @grant        GM_getResourceText
// @grant        GM_getResourceURL
// @nocompat     Chrome
// ==/UserScript==


class WebOCR {
    constructor() {
        this.MODEL_URL = GM_getResourceURL('CAPTCHA_MODEL');
        this.CHARSET = {
            '13': '6',
            '55': 'f',
            '209': 'p',
            '210': 'L',
            '297': 'Y',
            '306': 'w',
            '309': '3',
            '311': 'F',
            '320': 'm',
            '521': 'X',
            '598': 'G',
            '689': 'x',
            '782': 'i',
            '897': 'T',
            '901': 'N',
            '1072': 'v',
            '1150': 'c',
            '1204': 'B',
            '1503': 'n',
            '1849': 'Q',
            '1965': 'H',
            '2113': 'K',
            '2185': 'W',
            '2341': 'P',
            '2376': 'r',
            '2457': 'l',
            '2547': 'E',
            '2621': 'Z',
            '2714': 's',
            '2851': '2',
            '3073': 'z',
            '3128': 'D',
            '3157': 'O',
            '3606': '4',
            '4018': '1',
            '4102': 't',
            '4393': 'b',
            '4429': 'o',
            '4588': 'u',
            '4725': '9',
            '4730': 'j',
            '4733': '0',
            '4919': '8',
            '5223': '5',
            '5428': 'e',
            '5461': 'A',
            '5629': 'R',
            '5690': 'g',
            '5737': 'k',
            '5855': 'S',
            '6554': 'I',
            '6794': '7',
            '6810': 'd',
            '6887': 'V',
            '7216': 'J',
            '7266': 'a',
            '7412': 'h',
            '7576': 'q',
            '7712': 'U',
            '7844': 'M',
            '7877': 'y',
            '7961': 'C',
            '1151': 'c'
        };
        this.session = null;
        this.isLoadingModel = false;
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/"
    }

    async loadModel() {
        if (this.session || this.isLoadingModel) return;
        this.isLoadingModel = true;

        try {
            this.session = await ort.InferenceSession.create(this.MODEL_URL, {
                executionProviders: ['wasm']
            });
        } catch (e) {
            console.error('Failed to load ONNX model:', e);
            throw e;
        } finally {
            this.isLoadingModel = false;
        }
    }

    preprocessImage(imageElement) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        const targetHeight = 64;
        const originalWidth = imageElement.naturalWidth;
        const originalHeight = imageElement.naturalHeight;
        const targetWidth = Math.floor(originalWidth * (targetHeight / originalHeight));

        canvas.width = targetWidth;
        canvas.height = targetHeight;

        ctx.drawImage(imageElement, 0, 0, targetWidth, targetHeight);

        const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);
        const data = imageData.data;
        const inputData = new Float32Array(targetWidth * targetHeight);

        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            const grayscale = 0.299 * r + 0.587 * g + 0.114 * b;
            inputData[i / 4] = grayscale / 255.0;
        }

        return new ort.Tensor('float32', inputData, [1, 1, 64, targetWidth]);
    }

    decodeOutput(outputTensor, beamWidth = 3) {
        const outputData = outputTensor.data;
        const sequenceLength = outputTensor.dims[0];
        const numClasses = outputTensor.dims[2];

        // 初始化路径列表（[{ text, score, prev }]）
        let paths = [{
            text: '',
            score: 0,
            prev: -1
        }];

        for (let t = 0; t < sequenceLength; t++) {
            const nextPaths = [];

            for (const path of paths) {
                const probs = outputData.slice(t * numClasses, (t + 1) * numClasses);
                const sorted = Array.from(probs)
                    .map((p, i) => ({
                        prob: p,
                        index: i
                    }))
                    .sort((a, b) => b.prob - a.prob)
                    .slice(0, beamWidth);

                for (const {
                        prob,
                        index
                    }
                    of sorted) {
                    const char = this.CHARSET[index] || '';
                    const logProb = Math.log(prob + 1e-12); // 防止 log(0)

                    let newText = path.text;
                    if (index !== 0 && index !== path.prev) {
                        newText += char;
                    }

                    nextPaths.push({
                        text: newText,
                        score: path.score + logProb,
                        prev: index
                    });
                }
            }

            // 保留 top-K 路径（按分数排序）
            paths = nextPaths
                .sort((a, b) => b.score - a.score)
                .slice(0, beamWidth);
        }

        // 返回最高分路径文本
        return paths.length > 0 ? paths[0].text : '';
    }

    async classify(imageElement) {
        if (!this.session && !this.isLoadingModel) {
            await this.loadModel();
        }

        // 等待模型加载完成，最多等待10次，每次500ms
        let attempts = 0;
        const maxAttempts = 10;
        while (!this.session && attempts < maxAttempts) {
            await new Promise(resolve => setTimeout(resolve, 500));
            attempts++;
        }

        if (!this.session) {
            console.warn('Model not loaded after waiting.');
            return "";
        }

        const inputTensor = this.preprocessImage(imageElement);
        const feeds = {
            'input1': inputTensor
        };
        const results = await this.session.run(feeds);
        const outputTensor = Object.values(results)[0];

        const text = this.decodeOutput(outputTensor);
        return text;
    }
}


let Setting;

class CaptchaWrite {
    constructor() {
        this.Tip = this.AddTip();
        if (GM_listValues().indexOf("set") == -1) {
            GM_setValue("set", {});
            confirm("验证码填入\n初始化完毕!\n在将来的时间里将会在后台默默的为你\n自动识别页面是否存在验证码并填入。\n对于一些书写不规整的验证码页面请手动添加规则。");
        }

        Setting = GM_getValue("set");
        // 设置自动识别初始值
        var configSetKeys = {
            "autoIdentification": "true",
            "showHintCheck": "true",
            "warningTone": "false",
            "autoBlackList": "true",
            "hotKeyToImgResult": "false"
        };
        $.each(configSetKeys, function(key, val) {
            if (Setting[key] == undefined) {
                Setting[key] = val;
                GM_setValue("set", Setting);
            }
        });
    }

    // 恢复出厂设置
    clearSet() {
        var that = this;
        let res = confirm('您确认要恢复出厂设置吗？注意：清除后所有内容均需重新设置！');
        if (res == true) {
            GM_setValue("set", {});
        }
        return res;
    }

    //手动添加规则
    PickUp() {
        var that = this;
        var AddRule = {};
        var IdentifyResult = '';
        that.Hint('请对验证码图片点击右键！');
        $("canvas,img,input[type='image']").each(function() {
            $(this).on("contextmenu mousedown", function(e) { // 为了避免某些hook的拦截
                if (e.button != 2) { //不为右键则返回
                    return;
                }
                if (that.getCapFoowwLocalStorage("crabAddRuleLock") != null) {
                    return;
                }
                that.setCapFoowwLocalStorage("crabAddRuleLock", "lock", new Date().getTime() + 100); //100毫秒内只能1次
                var img = that.Aimed($(this));
                console.log('PickUp_Img:' + img);
                if ($(img).length != 1) {
                    that.Hint('验证码选择错误，该图片实际对应多个元素。')
                    return;
                }

                that.Hint('等待识别')
                IdentifyResult = that.Identify(img, function ManualRule(img, IdentifyResult) {
                    if (img && IdentifyResult) {
                        console.log('记录信息' + img + IdentifyResult);
                        AddRule['img'] = img;
                        $("img").each(function() {
                            $(this).off("click");
                            $(this).off("on");
                            $(this).off("load");
                        });
                        that.Hint('接下来请点击验证码输入框')
                        $("input").each(function() {
                            $(this).click(function() {
                                var input = that.Aimed($(this));
                                // console.log('PickUp_input' + input);
                                AddRule['input'] = input;
                                AddRule['path'] = window.location.href;
                                AddRule['title'] = document.title;
                                AddRule['host'] = window.location.host;
                                that.Write(IdentifyResult, input);
                                that.Hint('完成')
                                //移除事件
                                $("input").each(function() {
                                    $(this).off("click");
                                });
                                //添加信息
                                that.Query({
                                    "method": "captchaHostAdd",
                                    "data": AddRule
                                }, function(data) {
                                    writeResultIntervals[writeResultIntervals.length] = {
                                        "img": img,
                                        "input": input
                                    }
                                });
                                that.delCapFoowwLocalStorage(window.location.host);
                            });
                        });
                    }
                });


            });
        });
    }

    //创建提示元素
    AddTip() {
        var TipHtml = $("<div id='like996_identification'></div>").text("Text.");
        TipHtml.css({
            "background-color": "rgba(211,211,211,0.86)",
            "align-items": "center",
            "justify-content": "center",
            "position": "fixed",
            "color": "black",
            "top": "-5em",
            "height": "2em",
            "margin": "0em",
            "padding": "0em",
            "font-size": "1.2em",
            "width": "100%",
            "left": "0",
            "right": "0",
            "text-align": "center",
            "z-index": "9999999999999",
            "padding-top": "3px",
            display: 'none'

        });
        $("body").prepend(TipHtml);
        return TipHtml;
    }

    //展示提醒
    Hint(Content, Duration) {
        if (Setting["showHintCheck"] != "true") {
            return;
        }
        var that = this;

        that.Tip.stop(true, false).animate({
            top: '-5em'
        }, 300, function() {
            if (Setting["warningTone"] == "true") {
                Content += that.doWarningTone(Content)
            }
            Content += "<span style='color:red;float: right;margin-right: 20px;' onclick='document.getElementById(\"like996_identification\").remove()'>X</span>";
            that.Tip.show();
            that.Tip.html(Content);

        });
        that.Tip.animate({
            top: '0em'
        }, 500).animate({
            top: '0em'
        }, Duration ? Duration : 3000).animate({
            top: '-5em'
        }, 500, function() {
            that.Tip.hide();
        });
        return;
    }

    //查询规则
    Query(Json, callback) {
        var that = this;
        var QueryRule = '';
        var LocalStorageData = JSON.parse(localStorage.getItem(Json.method + "_" + Json.data.host));
        if (Json.method == 'captchaHostAdd') {
            localStorage.removeItem("captchaHostQuery_" + Json.data.host)
            // 永久保存规则
            localStorage.setItem("captchaHostQuery_" + Json.data.host, JSON.stringify(Json.data))
            callback(Json.data);
            return Json.data;
        }
        if (LocalStorageData != null) {
            console.log("存在本地缓存的验证码识别规则直接使用。")
            if (callback != null) {
                callback(LocalStorageData);
                return;
            } else {
                return {
                    code: 531,
                    data: [LocalStorageData]
                };
            }
        }

        return {
            "code": 533
        };
    }

    //开始识别
    Start() {
        //检查配置中是否有此网站
        var that = this;
        var Pathname = window.location.href;
        if (Setting["hotKeyToImgResult"] != "true") {
            writeResultInterval = setInterval(function() {
                that.WriteResultsInterval();
            }, 500);
        }

        var Rule = that.Query({
            "method": "captchaHostQuery",
            "data": {
                "host": window.location.host
            }
        })

        if (Rule.code == 531 || Rule.code == 532) {
            console.log('有规则执行规则' + Pathname);
            var data = Rule.data;
            for (var i = 0; i < data.length; i++) {
                writeResultIntervals[i] = data[i];
            }
            console.log('等待验证码图片出现');
        } else if (Rule.code == 533 && Setting["autoIdentification"] == "true") {
            console.log('新网站开始自动化验证码查找' + Pathname);
            var MatchList = that.AutoRules();
            if (MatchList.length) {
                console.log('检测到开始写入，并添加规则');
                for (i in MatchList) {
                    console.log(MatchList[i].img, MatchList[i].input);
                    $(MatchList[i].img).bind("error", function() {
                        that.addBadWeb(MatchList[i].img, MatchList[i].input);
                    });
                    that.WriteResults(MatchList[i].img, MatchList[i].input)
                }
            } else {}
        }
    }

    // 定时执行绑定验证码img操作
    WriteResultsInterval() {
        for (var i = 0; i < writeResultIntervals.length; i++) {
            var imgAddr = writeResultIntervals[i].img;
            var inputAddr = writeResultIntervals[i].input;
            if (document.querySelector(imgAddr) == null || document.querySelector(inputAddr) == null) {
                continue;
            }
            try {
                if (this.getCapFoowwLocalStorage("err_" + writeResultIntervals[i].img) == null) { // 写入识别规则之前，先判断她是否有错误
                    this.WriteResults(imgAddr, inputAddr);
                }
            } catch (e) {
                window.clearInterval(writeResultInterval);
                this.addBadWeb(imgAddr, inputAddr);
                return;
            }
        }
    }

    //解析
    Identify_Crab(img, Base, callback) {
        var that = this;

        var Results = that.getCapFoowwLocalStorage(Base.substring(Base.length - 32));
        if (Results != null) {
            if (callback.name != 'ManualRule') { // 不为手动直接返回结果
                return Results;
            }
        }

        that.setCapFoowwLocalStorage(Base.substring(Base.length - 32), "识别中..", new Date().getTime() + (9999999 * 9999999)); //同一个验证码只识别一次
        console.log("验证码变动，开始识别");

        const startTime = new Date().getTime();
        if (this.webocr === undefined) {
            this.webocr = new WebOCR();
        }
        const webocr = this.webocr;

        fetch(`data:image/png;base64,${Base}`)
            .then(res => res.blob())
            .then(blob => {
                return new Promise((resolve) => {
                    const imgEl = document.createElement("img");
                    imgEl.src = URL.createObjectURL(blob);
                    imgEl.onload = () => resolve(imgEl);
                });
            })
            .then(imgEl => webocr.classify(imgEl))
            .then(resultText => {
                Results = resultText;

                if (Results.length < 4) {
                    that.Hint('验证码识别结果可能错误，请刷新验证码尝试', 5000);
                } else {
                    that.Hint('验证码识别完成(耗时: ' + (new Date().getTime() - startTime) / 1000 + 's)', 500);
                }

                if (callback != null) {
                    if (callback.name === 'WriteRule') {
                        callback(Results);
                    } else if (callback.name === 'ManualRule') {
                        callback(img, Results);
                    }
                }

                return Results;
            })
            .catch(err => {
                that.Hint('验证码识别失败，请重试', 5000);
                console.error("OCR 失败:", err);
            });

        return Results;
    }

    //识别操作
    Identify(imgElement, callback) {
        var that = this;
        var imgObj = $(imgElement);
        if (!imgObj.is(":visible")) {
            console.log("验证码不可见，本次不识别");
            return;
        }
        try {
            imgObj = imgObj[0];
            var imgBase64;
            var imgSrc;
            var elementTagName = imgObj.tagName.toLowerCase();
            if (elementTagName === "img" || elementTagName === "input") {
                imgSrc = $(imgObj).attr("src");
            } else if (elementTagName === "div") {
                imgSrc = that.getElementStyle(imgObj)["backgroundImage"]
                if (imgSrc.trim().indexOf("data:image/") != -1) { //是base64格式得
                    imgSrc = imgSrc.match("(data:image/.*?;base64,.*?)[\"']")[1]
                }
            }

            if (imgSrc != undefined && imgSrc.indexOf("data:image/") == 0) {
                // 使用base64页面直显
                imgBase64 = imgSrc;
                // 兼容部分浏览器中replaceAll不存在
                while (imgBase64.indexOf("\n") != -1) {
                    imgBase64 = imgBase64.replace("\n", "");
                }
                // 解决存在url编码的换行问题
                while (imgBase64.indexOf("%0D%0A") != -1) {
                    imgBase64 = imgBase64.replace("%0D%0A", "");
                }
            } else if (imgSrc != undefined && (imgSrc.indexOf("http") == 0 || imgSrc.indexOf("//") == 0) && imgSrc.indexOf(window.location.protocol + "//" + window.location.host + "/") == -1) {
                // 跨域模式下单独获取src进行转base64
                var Results = that.getCapFoowwLocalStorage("验证码跨域识别锁：" + imgSrc);
                if (Results != null) {
                    return;
                }
                that.setCapFoowwLocalStorage("验证码跨域识别锁：" + imgSrc, "避免逻辑错误多次识别", new Date().getTime() + (9999999 * 9999999)); //同一个url仅识别一次

                GM_xmlhttpRequest({
                    url: imgSrc,
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json; charset=utf-8',
                        'path': window.location.href
                    },
                    responseType: "blob",
                    onload: obj => {
                        if (obj.status == 200) {
                            let blob = obj.response;
                            let fileReader = new FileReader();
                            fileReader.onloadend = (e) => {
                                let base64 = e.target.result;
                                $(imgObj).attr("src", base64);
                            };
                            fileReader.readAsDataURL(blob)
                        }
                    },
                    onerror: err => {
                        that.Hint('请求跨域图片异常');
                    }
                });
            } else {
                // 使用canvas进行图片转换
                imgBase64 = that.ConversionBase(imgElement).toDataURL("image/png");
            }

            var pastDate = imgBase64.replace(/.*,/, "").trim()
            if (pastDate.length < 255) {
                throw new Error("图片大小异常");
            }
        } catch (e) {
            if (callback.name == 'ManualRule') {
                that.Hint('跨域策略，请重新右键点击图片');
            }
            return;
        }

        that.Identify_Crab(imgElement, pastDate, callback);
    }

    //根据配置识别写入
    WriteResults(img, input) {
        var that = this;
        //创建一个触发操作
        if (document.querySelector(img) == null) {
            return;
        }

        document.querySelector(img).onload = function() {
            that.WriteResults(img, input)
        }

        this.Identify(img, function WriteRule(vcode) {
            that.Write(vcode, input)
        })

    }

    //写入操作
    Write(ResultsImg, WriteInput) {
        var that = this;
        WriteInput = document.querySelector(WriteInput);
        WriteInput.value = ResultsImg;
        if (typeof(InputEvent) !== 'undefined') {
            //使用 InputEvent 方法，主流浏览器兼容
            WriteInput.value = ResultsImg;
            WriteInput.dispatchEvent(new InputEvent("input")); //模拟事件
            that.fire(WriteInput, "change");
            that.fire(WriteInput, "blur");
            that.fire(WriteInput, "focus");
            that.fire(WriteInput, "keypress");
            that.fire(WriteInput, "keydown");
            that.fire(WriteInput, "keyup");
            that.fire(WriteInput, "select");
            that.fireForReact(WriteInput, "change");
            WriteInput.value = ResultsImg;
        } else if (KeyboardEvent) {
            //使用 KeyboardEvent 方法，ES6以下的浏览器方法
            WriteInput.dispatchEvent(new KeyboardEvent("input"));
        }
    }

    // 各类原生事件
    fire(element, eventName) {
        var event = document.createEvent("HTMLEvents");
        event.initEvent(eventName, true, true);
        element.dispatchEvent(event);
    }

    // 各类react事件
    fireForReact(element, eventName) {
        try {
            let env = new Event(eventName);
            element.dispatchEvent(env);
            var funName = Object.keys(element).find(p => Object.keys(element[p]).find(f => f.toLowerCase().endsWith(eventName)));
            if (!funName != undefined) {
                element[funName].onChange(env)
            }
        } catch (e) {
            console.log("各类react事件调用出错！")
        }

    }

    //转换图片为：canvas
    ConversionBase(img) {
        img = document.querySelector(img);
        var canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        var ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, img.width, img.height);
        return canvas;
    }

    //自动规则
    AutoRules() {
        var that = this;
        var MatchList = [];
        $("img").each(function() {
            var Randomcolor = "red";
            if ($(this).siblings("input").length == 1) {
                MatchList.push({
                    "img": that.Aimed($(this)),
                    "input": that.Aimed($(this).siblings("input"))
                })
                $(this).css("borderStyle", "solid").css("borderColor", Randomcolor).css("border-width", "4px");
                $(this).siblings("input").css("borderStyle", "solid").css("borderColor", Randomcolor);
            } else {
                if ($(this).prev().children("input").length == 1) {
                    MatchList.push({
                        "img": that.Aimed($(this)),
                        "input": that.Aimed($(this).prev().children("input"))
                    })
                    $(this).css("borderStyle", "solid").css("borderColor", Randomcolor).css("border-width", "4px");
                    $(this).prev().children("input").css("borderStyle", "solid").css("borderColor", Randomcolor);
                }
                if ($(this).next().children("input").length == 1) {
                    MatchList.push({
                        "img": that.Aimed($(this)),
                        "input": that.Aimed($(this).next().children("input"))
                    })
                    $(this).css("borderStyle", "solid").css("borderColor", Randomcolor).css("border-width", "4px");
                    $(this).next().children("input").css("borderStyle", "solid").css("borderColor", Randomcolor);
                }
            }
        });
        return MatchList;
    }

    //生成标识
    Aimed(Element) {
        // console.log('---根据元素创建配置信息---');
        Element = Element[0]
        var that = this;
        var ElementLocalName = Element.localName;
        var result;
        // 如果有vue的id，则直接返回
        var vueId = that.getDataV(Element);
        if (vueId != null) {
            result = ElementLocalName + "[" + vueId + "]";
            if ($(result).length == 1) {
                return result;
            }
        }
        // 如果有placeholder，则直接返回
        var placeholder = that.getPlaceholder(Element);
        if (placeholder != null) {
            result = ElementLocalName + "[" + placeholder + "]";
            if ($(result).length == 1) {
                return result;
            }
        }
        // 如果有alt，则直接返回
        var alt = that.getAlt(Element);
        if (alt != null) {
            result = ElementLocalName + "[" + alt + "]";
            if ($(result).length == 1) {
                return result;
            }
        }

        // 如果有name且只有一个，则直接返回
        var selectElement = that.getElementName(Element);
        if (selectElement != null) {
            return selectElement;
        }

        // 如果有src，且src后面无参数则直接返回
        var src = that.getSrc(Element);
        if (src != null && src.length < 200) {
            result = ElementLocalName + "[" + src + "]";
            if ($(result).length == 1) {
                return result;
            }
        }
        // 如果有onClick则直接返回
        var onClick = that.getOnClick(Element);
        if (onClick != null && onClick.length < 200) {
            result = ElementLocalName + "[" + onClick + "]";
            if ($(result).length == 1) {
                return result;
            }
        }
        var cssPath = that.geElementCssPath(Element);
        //alert(cssPath);
        if (cssPath != null && cssPath != "") {
            return cssPath;
        }

        var Symbol = (this.getElementId(Element) ? "#" : Element.className ? "." : false);
        var locationAddr;
        if (!Symbol) {
            locationAddr = that.Climb(Element.parentNode, ElementLocalName);
        } else {
            locationAddr = that.Climb(Element, ElementLocalName);
        }
        if ($(locationAddr).length == 1) {
            return locationAddr.trim();
        }


        if (confirm("当前元素无法自动选中，是否手动指定JsPath?\n(该功能为熟悉JavaScript的用户使用，若您不知道，请点击取消。)\n注意：如果该提示影响到您得操作了，关闭'自动查找验证码'功能即可！")) {
            result = prompt("请输入待选择元素的JsPath，例如：\n#app > div:nth-child(3) > div > input");
            try {
                if ($(result).length == 1) {
                    return result;
                }
            } catch (e) {}
        }

        that.Hint('该网站非标准web结构，暂时无法添加规则。')
        return null;

    }

    //判断元素id是否可信
    getElementId(element) {
        var id = element.id;
        if (id) {
            if (id.indexOf("exifviewer-img-") == -1) { // 对抗类似vue这种无意义id
                if (id.length < 40) { // 对抗某些会自动变换id的验证码
                    return true;
                }
            }
        }
        return false;
    }

    //爬层级
    Climb(Element, ElementLocalName, Joint = '') {
        var ElementType = (this.getElementId(Element) ? Element.id : Element.className ? Element.className.replace(/\s/g, ".") : false);
        var Symbol = (this.getElementId(Element) ? "#" : Element.className ? "." : false);
        var Address;
        if (ElementType && ElementLocalName == Element.localName) {
            Address = ElementLocalName + Symbol + ElementType;
        } else {
            Address = "";
            if (Symbol != false) {
                Address = Address + Symbol;
            }
            if (ElementType != false) {
                Address = Address + ElementType;
            }
            Address = ' ' + ElementLocalName
        }
        if ($(Address).length == 1) {
            return Address + ' ' + Joint;
        } else {
            Joint = this.Climb($(Element).parent()[0], $(Element).parent()[0].localName, Address + ' ' + Joint)
            return Joint;
        }
    }

    // 获取vue的data-v-xxxx
    getDataV(element) {
        var elementKeys = element.attributes;
        if (elementKeys == null) {
            return null;
        }
        for (var i = 0; i < elementKeys.length; i++) {
            var key = elementKeys[i].name;
            if (key.indexOf("data-v-") != -1) {
                return key;
            }
        }
        return null;
    }

    // 获取placeholder="验证码"
    getPlaceholder(element) {
        var elementKeys = element.attributes;
        if (elementKeys == null) {
            return null;
        }
        for (var i = 0; i < elementKeys.length; i++) {
            var key = elementKeys[i].name.toLowerCase();
            if (key == "placeholder" && elementKeys[i].value != "") {
                return elementKeys[i].name + "='" + elementKeys[i].value + "'";
            }
        }
        return null;
    }

    // 获取alt="kaptcha"
    getAlt(element) {
        var elementKeys = element.attributes;
        if (elementKeys == null) {
            return null;
        }
        for (var i = 0; i < elementKeys.length; i++) {
            var key = elementKeys[i].name.toLowerCase();
            if (key == "alt") {
                return elementKeys[i].name + "='" + elementKeys[i].value + "'";
            }
        }
        return null;
    }

    // 获取src="http://xxx.com"
    getSrc(element) {
        var elementKeys = element.attributes;
        if (elementKeys == null) {
            return null;
        }
        for (var i = 0; i < elementKeys.length; i++) {
            var key = elementKeys[i].name.toLowerCase();
            var value = elementKeys[i].value;
            if (key == "src" && value.indexOf("data:image") != 0) {
                var idenIndex = value.indexOf("?");
                if (idenIndex != -1) {
                    value = value.substring(0, idenIndex + 1);
                }
                return elementKeys[i].name + "^='" + value + "'";
            }
        }
        return null;
    }

    // 判断name是否只有一个
    getElementName(element) {
        var elementName = element.name;
        if (elementName == null || elementName == "") {
            return null;
        }
        var selectElement = element.localName + "[name='" + elementName + "']";
        if ($(selectElement).length == 1) {
            return selectElement;
        }
        return null;
    }

    // 判断OnClick是否只有一个
    getOnClick(element) {
        var elementKeys = element.attributes;
        if (elementKeys == null) {
            return null;
        }
        for (var i = 0; i < elementKeys.length; i++) {
            var key = elementKeys[i].name.toLowerCase();
            var value = elementKeys[i].value;
            if (key == "onclick") {
                var idenIndex = value.indexOf("(");
                if (idenIndex != -1) {
                    value = value.substring(0, idenIndex + 1);
                }
                return elementKeys[i].name + "^='" + value + "'";
            }
        }
        return null;
    }

    // 操作webStorage 增加缓存，减少对服务端的请求
    setCapFoowwLocalStorage(key, value, ttl_ms) {
        var data = {
            value: value,
            expirse: new Date(ttl_ms).getTime()
        };
        sessionStorage.setItem(key, JSON.stringify(data));
    }

    getCapFoowwLocalStorage(key) {
        var data = JSON.parse(sessionStorage.getItem(key));
        if (data !== null) {
            if (data.expirse != null && data.expirse < new Date().getTime()) {
                sessionStorage.removeItem(key);
            } else {
                return data.value;
            }
        }
        return null;
    }

    delCapFoowwLocalStorage(key) {
        window.sessionStorage.removeItem(key);
    }

    // 添加识别错误黑名单
    addBadWeb(img, input) {
        if (Setting["autoBlackList"] == "false") {
            return;
        }
        this.Hint('识别过程中发生错误，已停止识别此网站！（若验证码消失，请刷新网站）', 10000);
        this.setCapFoowwLocalStorage("err_" + img, "可能存在跨域等问题停止操作它", new Date().getTime() + (1000 * 1000));
        this.delCapFoowwLocalStorage("captchaHostQuery_" + window.location.host);
    }

    // 播放音频朗读
    doWarningTone(body) {
        if (body.indexOf("，")) {
            body = body.split("，")[0];
        }
        if (body.indexOf(",")) {
            body = body.split(",")[0];
        }
        if (body.indexOf("!")) {
            body = body.split("!")[0];
        }
        var zhText = encodeURI(body);
        var text = "<audio autoplay='autoplay'>" +
            "<source src='https://tts.youdao.com/fanyivoice?le=cn&keyfrom=speaker-target&word=" + zhText + "' type='audio/mpeg'>" +
            "<embed height='0' width='0' src='https://tts.youdao.com/fanyivoice?le=cn&keyfrom=speaker-target&word=" + zhText + "'>" +
            "</audio>";
        return text;
    }

    // 获取元素的全部样式
    getElementStyle(element) {
        if (window.getComputedStyle) {
            return window.getComputedStyle(element, null);
        } else {
            return element.currentStyle;
        }
    }

    // 获取元素的cssPath选择器
    geElementCssPath(element) {
        if (!(element instanceof Element)) return;
        var path = [];
        while (element.nodeType === Node.ELEMENT_NODE) {
            var selector = element.nodeName.toLowerCase();
            if (element.id && element.id.indexOf("exifviewer-img-") == -1) {
                selector += "#" + element.id;
                path.unshift(selector);
                break;
            } else {
                var sib = element,
                    nth = 1;
                while ((sib = sib.previousElementSibling)) {
                    if (sib.nodeName.toLowerCase() == selector) nth++;
                }
                if (nth != 1) selector += ":nth-of-type(" + nth + ")";
            }
            path.unshift(selector);
            element = element.parentNode;
        }
        return path.join(" > ");
    }


}

//所有验证码img的对象数组
var writeResultIntervals = [];

//定时执行验证码绑定操作定时器
var writeResultInterval;


function closeButton() {
    const closebtn = document.createElement("div");
    closebtn.innerHTML = " × ";
    closebtn.style.position = "absolute";
    closebtn.style.top = "10px";
    closebtn.style.right = "10px";
    closebtn.style.cursor = "pointer";
    closebtn.style.fontWeight = 900;
    closebtn.style.fontSize = "larger";
    closebtn.setAttribute("onclick", "CKTools.modal.hideModal()");
    return closebtn;
}

async function GUISettings() {
    if (CKTools.modal.isModalShowing()) {
        CKTools.modal.hideModal();
        await wait(300);
    }
    const menuList = [{
            name: 'autoIdentification',
            title: '自动查找无规则验证码',
            hintOpen: '已开启自动查找验证码功能，请刷新网页',
            hintClose: '已关闭自动查找验证码功能，遇到新网站请自行手动添加规则!',
            desc: '对于未添加规则的页面，将自动查找页面上的验证码，有找错的可能。',
            openVul: 'true',
            closeVul: 'false'
        },
        {
            name: 'showHintCheck',
            title: '提示信息',
            hintOpen: '提示功能已开启！',
            hintClose: '提示功能已关闭，再次开启前将无任何提示！',
            desc: '关闭前请确保已知晓插件的使用流程！',
            openVul: 'true',
            closeVul: 'false'
        },
        {
            name: 'warningTone',
            title: '提示音',
            hintOpen: '提示音功能已开启！',
            hintClose: '提示音功能已关闭！',
            desc: '自动朗读提示信息中的文字！',
            openVul: 'true',
            closeVul: 'false'
        },
        {
            name: 'autoBlackList',
            title: '识别崩溃自动拉黑网站',
            hintOpen: '崩溃自动拉黑网站功能已开启！',
            hintClose: '崩溃自动拉黑网站功能已关闭！',
            desc: '遇到跨域或其他错误导致验证码无法加载时自动将网站加到黑名单中。',
            openVul: 'true',
            closeVul: 'false'
        },
        {
            name: 'hotKeyToImgResult',
            title: '快捷键查找验证码',
            hintOpen: '请直接按下您需要设置的快捷键！设置快捷键前请确保当前页面能够自动识别否则先手动添加规则！',
            hintClose: '快捷键查找验证码已关闭！',
            desc: '先手动添加规则后再开启，开启后将停止自动识别，仅由快捷键识别！',
            openVul: 'wait',
            closeVul: 'false'
        },
        {
            name: 'clearSet',
            type: 'button',
            title: '恢复出厂设置',
            hintOpen: '已成功恢复出厂设置刷新页面即可生效',
            desc: '清除所有设置！',
            doWork: 'crabCaptcha.clearSet()'
        },
    ]
    CKTools.modal.openModal("万能验证码自动输入-更多设置（点击切换）", await CKTools.domHelper("div", async container => {
        container.appendChild(closeButton());
        container.style.alignItems = "stretch";
        for (var i = 0; i < menuList.length; i++) {
            container.appendChild(await CKTools.domHelper("li", async list => {
                list.classList.add("showav_menuitem");
                if (menuList[i].type == 'button') {
                    list.appendChild(await CKTools.domHelper("label", label => {
                        label.id = menuList[i].name + "Tip";
                        label.value = i;
                        label.setAttribute('doWork', menuList[i].doWork);
                        label.addEventListener("click", e => {
                            if (eval($(e.target).attr("doWork"))) {
                                crabCaptcha.Hint(menuList[e.target.value].hintOpen);
                            }
                        })
                        label.innerHTML = menuList[i].title;
                    }));
                } else {
                    list.appendChild(await CKTools.domHelper("input", input => {
                        input.type = "checkbox";
                        input.id = menuList[i].name;
                        input.name = menuList[i].name;
                        input.value = i;
                        input.style.display = "none";
                        input.checked = Setting[menuList[i].name] == 'true';
                        input.addEventListener("change", e => {
                            var i = e.target.value;
                            const label = document.querySelector("#" + menuList[i].name + "Tip");
                            if (!label) return;
                            if (input.checked) {
                                label.innerHTML = "<b>[已开启]</b> " + menuList[i].title;
                                Setting[menuList[i].name] = menuList[i].openVul;
                                GM_setValue("set", Setting);
                                crabCaptcha.Hint(menuList[i].hintOpen);
                            } else {
                                label.innerHTML = "<span>[已关闭]</span>" + menuList[i].title;
                                Setting[menuList[i].name] = menuList[i].closeVul;
                                GM_setValue("set", Setting);
                                crabCaptcha.Hint(menuList[i].hintClose);
                            }
                        })
                    }));
                    list.appendChild(await CKTools.domHelper("label", label => {
                        label.id = menuList[i].name + "Tip";
                        label.setAttribute('for', menuList[i].name);
                        if (Setting[menuList[i].name] == 'true') {
                            label.innerHTML = "<b>[已开启]</b>" + menuList[i].title;
                        } else {
                            label.innerHTML = "<span>[已关闭]</span>" + menuList[i].title;
                        }
                    }));
                }
                list.appendChild(await CKTools.domHelper("div", div => {
                    div.style.paddingLeft = "20px";
                    div.style.color = "#919191";
                    div.innerHTML = "说明：" + menuList[i].desc;;
                }));
                list.style.lineHeight = "2em";
            }))
        }
        container.appendChild(await CKTools.domHelper("div", async btns => {
            btns.style.display = "flex";
            btns.style.alignItems = "flex-end";
            btns.appendChild(await CKTools.domHelper("button", btn => {
                btn.className = "CKTOOLS-toolbar-btns";
                btn.innerHTML = "关闭";
                btn.style.background = "#ececec";
                btn.style.color = "black";
                btn.onclick = e => {
                    CKTools.addStyle(``, "showav_lengthpreviewcss", "update");
                    CKTools.modal.hideModal();
                    wait(300).then(() => GUISettings());
                }
            }))
        }))
    }));
}

var crabCaptcha = new CaptchaWrite();
(function() {
    const resourceList = [{
        name: 'cktools',
        type: 'js'
    }]

    function applyResource() {
        resloop: for (let res of resourceList) {
            if (!document.querySelector("#" + res.name)) {
                let el;
                switch (res.type) {
                    case 'js':
                    case 'rawjs':
                        el = document.createElement("script");
                        break;
                    case 'css':
                    case 'rawcss':
                        el = document.createElement("style");
                        break;
                    default:
                        console.log('Err:unknown type', res);
                        continue resloop;
                }
                el.id = res.name;
                el.innerHTML = res.type.startsWith('raw') ? res.content : GM_getResourceText(res.name);
                document.head.appendChild(el);
            }
        }
    }

    applyResource();
    GM_registerMenuCommand('手动添加规则', function() {
        crabCaptcha.PickUp();
    }, 'a');

    GM_registerMenuCommand('更多设置', function() {
        GUISettings();
    }, 'u');
    crabCaptcha.Start();
    CKTools.addStyle(`
    #CKTOOLS-modal{
        width: fit-content!important;
        max-width: 80%!important;
    }
    .CKTOOLS-modal-content li label b {
        color: green!important;
    }
    .CKTOOLS-modal-content li label span {
        color: red!important;
    }
    .showav_menuitem{
        line-height: 2em;
        width: 100%;
        transition: all .3s;
        cursor: pointer;
    }
    .showav_menuitem:hover{
        transform: translateX(6px);
    }
    .showav_menuitem>label{
        font-weight: bold;
        font-size: large;
        display: block;
    }
    `, 'showav_dragablecss', "unique", document.head);

    CKTools.addStyle(`
    #CKTOOLS-modal li, #CKTOOLS-modal ul{
        list-style: none !important;
    }
    `, 'showav_css_patch', 'unique', document.head);
})();

// 监控热键
document.onkeydown = function() {
    if (Setting["hotKeyToImgResult"] == "false") {
        return;
    }
    var keyCodeName = {
        "91": "command",
        "96": "0",
        "97": "1",
        "98": "2",
        "99": "3",
        "100": "4",
        "101": "5",
        "102": "6",
        "103": "7",
        "104": "8",
        "105": "9",
        "106": "*",
        "107": "+",
        "108": "回车",
        "109": "-",
        "110": ".",
        "111": "/",
        "112": "F1",
        "113": "F2",
        "114": "F3",
        "115": "F4",
        "116": "F5",
        "117": "F6",
        "118": "F7",
        "119": "F8",
        "120": "F9",
        "121": "F10",
        "122": "F11",
        "123": "F12",
        "8": "BackSpace",
        "9": "Tab",
        "12": "Clear",
        "13": "回车",
        "16": "Shift",
        "17": "Control",
        "18": "Alt",
        "20": "Cape Lock",
        "27": "Esc",
        "32": "空格",
        "33": "Page Up",
        "34": "Page Down",
        "35": "End",
        "36": "Home",
        "37": "←",
        "38": "↑",
        "39": "→",
        "40": "↓",
        "45": "Insert",
        "46": "Delete",
        "144": "Num Lock",
        "186": ";",
        "187": "=",
        "188": ",",
        "189": "-",
        "190": ".",
        "191": "/",
        "192": "`",
        "219": "[",
        "220": "\\",
        "221": "]",
        "222": "'",
        "65": "A",
        "66": "B",
        "67": "C",
        "68": "D",
        "69": "E",
        "70": "F",
        "71": "G",
        "72": "H",
        "73": "I",
        "74": "J",
        "75": "K",
        "76": "L",
        "77": "M",
        "78": "N",
        "79": "O",
        "80": "P",
        "81": "Q",
        "82": "R",
        "83": "S",
        "84": "T",
        "85": "U",
        "86": "V",
        "87": "W",
        "88": "X",
        "89": "Y",
        "90": "Z",
        "48": "0",
        "49": "1",
        "50": "2",
        "51": "3",
        "52": "4",
        "53": "5",
        "54": "6",
        "55": "7",
        "56": "8",
        "57": "9"
    };
    var a = window.event.keyCode;
    if (Setting["hotKeyToImgResult"] == "wait" && a != undefined) {
        var keyName = keyCodeName[a + ""] == undefined ? a : keyCodeName[a + ""];
        crabCaptcha.Hint('快捷键设置成功当前快捷键为:' + keyName + "，重新打开页面生效！");
        Setting["hotKeyToImgResult"] = "true";
        Setting["hotKey"] = a;
        GM_setValue("set", Setting);
        clearInterval(writeResultInterval);
    } else {
        if (a == Setting["hotKey"]) {
            crabCaptcha.WriteResultsInterval();
            crabCaptcha.Hint("开始快捷键识别验证码,在当前页面刷新之前新的验证码将自动识别！");
        }
    }
}