import DefaultTheme from 'vitepress/theme'
import { h, onMounted, ref } from 'vue'
import './style.css'

const AUTH_PASSWORD = 'demo123' // 在这里修改你的密码

function checkAuth() {
  if (typeof window !== 'undefined') {
    return sessionStorage.getItem('auth') === 'true'
  }
  return true
}

function showPasswordOverlay() {
  const overlay = document.createElement('div')
  overlay.id = 'password-overlay'
  overlay.innerHTML = `
    <div id="password-card">
      <h1>请输入密码</h1>
      <input type="password" id="password-input" placeholder="输入密码">
      <button id="password-btn">确认</button>
      <div id="password-error">密码错误，请重试</div>
    </div>
  `

  const style = document.createElement('style')
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
    #password-input {
      width: 200px;
      padding: 12px 16px;
      border: 2px solid #333;
      border-radius: 8px;
      background: #222;
      color: white;
      font-size: 1rem;
      margin-bottom: 16px;
    }
    #password-input:focus {
      outline: none;
      border-color: #4a9eff;
    }
    #password-btn {
      padding: 12px 40px;
      background: #4a9eff;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 500;
      cursor: pointer;
    }
    #password-btn:hover {
      background: #5aa8ff;
    }
    #password-error {
      color: #ff4757;
      margin-top: 12px;
      font-size: 0.85rem;
      display: none;
    }
  `

  document.head.appendChild(style)
  document.body.appendChild(overlay)

  const btn = document.getElementById('password-btn')
  const input = document.getElementById('password-input')
  const error = document.getElementById('password-error')

  function handlePassword() {
    if (input.value === AUTH_PASSWORD) {
      sessionStorage.setItem('auth', 'true')
      overlay.style.display = 'none'
    } else {
      error.style.display = 'block'
    }
  }

  btn.addEventListener('click', handlePassword)
  input.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handlePassword()
  })
}

const PasswordWrapper = {
  setup() {
    onMounted(() => {
      if (!checkAuth()) {
        showPasswordOverlay()
      }
    })
    return () => h(DefaultTheme.Layout, null, {})
  }
}

export default {
  extends: DefaultTheme,
  Layout: PasswordWrapper,
  enhanceApp(ctx) {
    DefaultTheme.enhanceApp(ctx)
  }
}
