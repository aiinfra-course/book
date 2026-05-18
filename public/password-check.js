(function() {
  const PASSWORD = 'demo123'; // 设置你的密码
  
  // 等待 DOM 加载完成后再执行
  function initPasswordProtection() {
    if (sessionStorage.getItem('auth') === 'true') {
      return; // 已经认证过，不需要显示密码框
    }
    
    const overlay = document.createElement('div');
    overlay.id = 'password-overlay';
    overlay.innerHTML = `
      <div id="password-card">
        <h1>请输入密码</h1>
        <input type="password" id="password" placeholder="输入密码">
        <button onclick="checkPassword()">确认</button>
        <div id="error">密码错误，请重试</div>
      </div>
    `;
    
    const style = document.createElement('style');
    style.textContent = `
      #password-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: #1a1a2e;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 999999;
      }
      #password-card {
        text-align: center;
        padding: 40px;
        background: #16213e;
        border-radius: 12px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
      }
      #password-card h1 {
        margin-bottom: 20px;
        font-size: 1.5rem;
        color: white;
      }
      #password {
        width: 200px;
        padding: 12px 16px;
        border: 2px solid #333;
        border-radius: 8px;
        background: #222;
        color: white;
        font-size: 1rem;
        margin-bottom: 16px;
      }
      #password:focus {
        outline: none;
        border-color: #4a9eff;
      }
      #password-card button {
        padding: 12px 40px;
        background: #4a9eff;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 500;
        cursor: pointer;
      }
      #password-card button:hover {
        background: #5aa8ff;
      }
      #error {
        color: #ff4757;
        margin-top: 12px;
        font-size: 0.85rem;
        display: none;
      }
    `;
    
    document.head.appendChild(style);
    document.body.appendChild(overlay);
    
    window.checkPassword = function() {
      const input = document.getElementById('password').value;
      if (input === PASSWORD) {
        sessionStorage.setItem('auth', 'true');
        document.getElementById('password-overlay').style.display = 'none';
      } else {
        document.getElementById('error').style.display = 'block';
      }
    };
    
    document.getElementById('password').addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        checkPassword();
      }
    });
    
    // 聚焦输入框
    setTimeout(() => {
      document.getElementById('password').focus();
    }, 100);
  }
  
  // 等待页面加载完成
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPasswordProtection);
  } else {
    initPasswordProtection();
  }
})();
