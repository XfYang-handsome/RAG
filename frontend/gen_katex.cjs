const code = `return new Promise(resolve => {
  const app = document.querySelector('#app');
  const v = app.__vue_app__;
  const p = v.config.globalProperties.$pinia;
  const c = p._s.get('chat');
  const bs = '\\\\';
  const formula = '不定积分 ' + bs + '(' + bs + 'int ' + bs + 'frac{1}{x^5+1}' + bs + ', dx' + bs + ') 的结果。';
  c.messages.push({id: 999, role: 'user', content: '求积分', pending: false, reasoning: '', citations: [], agentTrace: [], status: '', warning: '', errorText: ''});
  c.messages.push({id: 1000, role: 'assistant', content: formula, pending: false, reasoning: '', citations: [], agentTrace: [], status: '', warning: '', errorText: ''});
  setTimeout(() => {
    const k = document.querySelector('.katex');
    if (!k) { resolve('NO .katex element'); return; }
    const khtml = k.querySelector('.katex-html');
    const cs = khtml ? window.getComputedStyle(khtml) : null;
    // 检查 KaTeX CSS 是否加载：katex.min.css 里有 .katex { font: normal 1.21em KaTeX_Main }
    const font = cs ? cs.fontFamily : 'no-cs';
    // 检查样式表
    let hasKatexCss = false;
    for (const sheet of document.styleSheets) {
      try {
        const rules = sheet.cssRules || [];
        for (const r of rules) {
          if (r.selectorText && r.selectorText.includes('.katex')) { hasKatexCss = true; break; }
        }
      } catch (e) {}
      if (hasKatexCss) break;
    }
    // 检查 .katex-html 的 display
    const display = cs ? cs.display : 'no-cs';
    resolve('katex exists, fontFamily=' + font + ', .katex-html display=' + display + ', katexCssLoaded=' + hasKatexCss);
  }, 800);
});`;
const b64 = Buffer.from(code).toString('base64');
const cmd = `async page => { const r = await page.evaluate(() => { const s = atob('${b64}'); return (new Function(s))(); }); return r; }`;
console.log(cmd);
