<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FitLife App</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

  :root {
    --yellow: #E8FF47;
    --yellow-soft: #F7FFB5;
    --yellow-dark: #C8E000;
    --green: #1A3A2A;
    --green-mid: #2D5A3D;
    --green-light: #4CAF6F;
    --green-accent: #4CAF6F;
    --bg: #F7F8F5;
    --bg-soft: #FFFFFF;
    --bg-card: #FFFFFF;
    --bg-elevated: #FAFBF7;
    --text: #1A2A20;
    --text-soft: #4A5A50;
    --text-muted: #8A9890;
    --border: #E2E8DE;
    --border-strong: #CDD6C8;
    --red: #E04848;
    --blue: #4A90E2;
    --radius: 16px;
    --radius-sm: 10px;
    --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    --shadow-sm: 0 2px 8px rgba(26,42,32,0.04);
    --shadow-md: 0 6px 20px rgba(26,42,32,0.08);
    --shadow-lg: 0 16px 40px rgba(26,42,32,0.12);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body { font-family: 'DM Sans', sans-serif; background: #DFE5DA; display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; color: var(--text); }

  .phone-frame { width: 390px; height: 844px; background: var(--bg); border-radius: 48px; overflow: hidden; position: relative; box-shadow: var(--shadow-lg), inset 0 0 0 2px rgba(26,42,32,0.06); }

  .screen { width: 100%; height: 100%; overflow-y: auto; overflow-x: hidden; scrollbar-width: none; position: absolute; top: 0; left: 0; display: none; flex-direction: column; background: var(--bg); }
  .screen::-webkit-scrollbar { display: none; }
  .screen.active { display: flex; }

  /* STATUS BAR */
  .status-bar { display: flex; justify-content: space-between; align-items: center; padding: 14px 28px 0; font-size: 13px; font-weight: 600; color: var(--text); flex-shrink: 0; }
  .status-icons { display: flex; gap: 6px; align-items: center; }

  /* STEP BREADCRUMB */
  .step-trail { padding: 10px 24px; font-size: 11px; color: var(--text-muted); border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 6px; flex-wrap: wrap; flex-shrink: 0; background: var(--bg-soft); }
  .step-trail span.current { color: var(--green); font-weight: 700; }
  .step-trail .sep { color: var(--border-strong); }

  /* TOP NAV */
  .top-nav { display: flex; align-items: center; justify-content: space-between; padding: 14px 24px 0; flex-shrink: 0; }
  .top-nav-title { font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; color: var(--text); }
  .back-btn { width: 36px; height: 36px; background: var(--bg-soft); border: 1px solid var(--border); border-radius: 10px; display: flex; align-items: center; justify-content: center; cursor: pointer; color: var(--text); font-size: 18px; transition: var(--transition); }
  .back-btn:hover { border-color: var(--green); background: var(--bg-elevated); }

  .screen-content { padding: 20px 24px; flex: 1; overflow-y: auto; }

  /* BUTTONS */
  .btn-primary { width: 100%; padding: 16px; background: var(--green); color: white; border: none; border-radius: var(--radius-sm); font-size: 16px; font-weight: 700; font-family: 'DM Sans', sans-serif; cursor: pointer; transition: var(--transition); letter-spacing: -0.3px; box-shadow: var(--shadow-sm); }
  .btn-primary:hover { background: var(--green-mid); transform: translateY(-1px); box-shadow: var(--shadow-md); }
  .btn-yellow { width: 100%; padding: 16px; background: var(--yellow); color: var(--green); border: none; border-radius: var(--radius-sm); font-size: 16px; font-weight: 700; font-family: 'DM Sans', sans-serif; cursor: pointer; transition: var(--transition); }
  .btn-yellow:hover { background: var(--yellow-dark); transform: translateY(-1px); }
  .btn-secondary { width: 100%; padding: 14px; background: var(--bg-soft); color: var(--text); border: 1.5px solid var(--border); border-radius: var(--radius-sm); font-size: 15px; font-weight: 600; font-family: 'DM Sans', sans-serif; cursor: pointer; transition: var(--transition); margin-top: 12px; }
  .btn-secondary:hover { border-color: var(--green); color: var(--green); }
  .btn-danger { width: 100%; padding: 14px; background: rgba(224,72,72,0.08); color: var(--red); border: 1.5px solid rgba(224,72,72,0.25); border-radius: var(--radius-sm); font-size: 15px; font-weight: 600; font-family: 'DM Sans', sans-serif; cursor: pointer; transition: var(--transition); margin-top: 12px; }

  /* SPLASH */
  #screen-splash { background: linear-gradient(180deg, var(--bg) 0%, #EFF3EB 100%); justify-content: center; align-items: center; gap: 24px; }
  .splash-logo { width: 88px; height: 88px; background: var(--green); border-radius: 24px; display: flex; align-items: center; justify-content: center; font-size: 38px; animation: pulse 2s infinite; box-shadow: var(--shadow-md); position: relative; }
  .splash-logo::after { content: ''; position: absolute; bottom: -8px; right: -8px; width: 26px; height: 26px; background: var(--yellow); border-radius: 50%; border: 3px solid var(--bg); }
  @keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.05); } }
  .splash-title { font-family: 'Space Grotesk', sans-serif; font-size: 42px; font-weight: 700; color: var(--text); letter-spacing: -1px; }
  .splash-title span { color: var(--green); }
  .splash-sub { color: var(--text-muted); font-size: 16px; }

  /* DECISION SCREEN */
  .decision-screen { display: flex; flex-direction: column; align-items: center; justify-content: center; flex: 1; padding: 40px 28px; gap: 32px; text-align: center; }
  .decision-diamond {
    width: 220px; height: 220px;
    background: var(--bg-soft);
    border: 2px solid var(--green);
    transform: rotate(45deg);
    display: flex; align-items: center; justify-content: center;
    border-radius: 16px;
    position: relative;
    box-shadow: var(--shadow-md);
  }
  .decision-diamond-inner { transform: rotate(-45deg); padding: 20px; text-align: center; }
  .decision-diamond-inner .q-icon { font-size: 32px; margin-bottom: 8px; }
  .decision-diamond-inner .q-text { font-family: 'Space Grotesk', sans-serif; font-size: 16px; font-weight: 700; color: var(--text); line-height: 1.3; }
  .decision-actions { display: flex; gap: 12px; width: 100%; }
  .decision-actions button { flex: 1; }
  .decision-context { font-size: 13px; color: var(--text-soft); line-height: 1.5; max-width: 280px; }

  /* AUTH HERO */
  .auth-hero { background: linear-gradient(170deg, #FFFFFF 0%, #EFF3EB 100%); padding: 60px 28px 40px; position: relative; overflow: hidden; border-bottom: 1px solid var(--border); }
  .auth-hero::before { content: ''; position: absolute; top: -40px; right: -40px; width: 180px; height: 180px; background: var(--yellow); border-radius: 50%; opacity: 0.45; }
  .auth-hero::after { content: ''; position: absolute; bottom: -30px; left: -30px; width: 100px; height: 100px; background: var(--green); border-radius: 50%; opacity: 0.08; }
  .auth-hero-tag { display: inline-block; background: var(--green); color: white; font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 20px; margin-bottom: 16px; letter-spacing: 1px; text-transform: uppercase; position: relative; z-index: 1; }
  .auth-hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: 36px; font-weight: 700; color: var(--text); line-height: 1.1; margin-bottom: 10px; position: relative; z-index: 1; }
  .auth-hero h1 em { color: var(--green); font-style: normal; background: var(--yellow); padding: 0 8px; border-radius: 6px; }
  .auth-hero p { color: var(--text-soft); font-size: 15px; position: relative; z-index: 1; }

  /* AUTH CHOICE */
  .choice-card { background: var(--bg-card); border: 1.5px solid var(--border); border-radius: var(--radius); padding: 22px; cursor: pointer; transition: var(--transition); display: flex; align-items: center; gap: 16px; margin-bottom: 12px; box-shadow: var(--shadow-sm); }
  .choice-card:hover { border-color: var(--green); background: var(--bg-elevated); transform: translateY(-2px); box-shadow: var(--shadow-md); }
  .choice-card-icon { width: 56px; height: 56px; background: var(--yellow); border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 28px; flex-shrink: 0; }
  .choice-card-info { flex: 1; }
  .choice-card-title { font-family: 'Space Grotesk', sans-serif; font-size: 18px; font-weight: 700; color: var(--text); margin-bottom: 4px; }
  .choice-card-sub { font-size: 13px; color: var(--text-muted); }
  .choice-card-arrow { font-size: 20px; color: var(--green); }

  /* FORM */
  .form-group { margin-bottom: 16px; }
  .form-label { font-size: 12px; font-weight: 600; color: var(--text-soft); letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 8px; display: block; }
  .form-input { width: 100%; padding: 14px 16px; background: var(--bg-soft); border: 1.5px solid var(--border); border-radius: var(--radius-sm); color: var(--text); font-size: 15px; font-family: 'DM Sans', sans-serif; transition: var(--transition); outline: none; }
  .form-input:focus { border-color: var(--green); background: white; box-shadow: 0 0 0 3px rgba(26,58,42,0.06); }
  .form-input::placeholder { color: var(--text-muted); }

  /* STEP TITLE */
  .step-title { font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 700; color: var(--text); margin-bottom: 8px; }
  .step-sub { color: var(--text-soft); font-size: 14px; margin-bottom: 24px; line-height: 1.5; }

  /* MAIL CONFIRMATION */
  .mail-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; text-align: center; margin-bottom: 24px; box-shadow: var(--shadow-sm); position: relative; overflow: hidden; }
  .mail-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, var(--green), var(--yellow)); }
  .mail-icon { font-size: 56px; margin-bottom: 16px; }
  .mail-status { font-size: 14px; color: var(--text-muted); margin-bottom: 6px; }
  .mail-addr { font-size: 15px; font-weight: 700; color: var(--green); word-break: break-all; }

  /* T&C */
  .terms-box { background: var(--bg-soft); border-radius: var(--radius-sm); padding: 16px; margin-bottom: 24px; max-height: 280px; overflow-y: auto; border: 1px solid var(--border); }
  .terms-box h4 { color: var(--text); font-size: 13px; font-weight: 700; margin: 12px 0 6px; }
  .terms-box h4:first-child { margin-top: 0; }
  .terms-box p { color: var(--text-soft); font-size: 12px; line-height: 1.6; }
  .terms-checkbox { display: flex; gap: 10px; align-items: flex-start; margin-bottom: 16px; cursor: pointer; }
  .terms-checkbox input { margin-top: 3px; accent-color: var(--green); }
  .terms-checkbox label { font-size: 13px; color: var(--text); cursor: pointer; }

  /* PLAN */
  .plan-card { background: var(--bg-card); border: 2px solid var(--border); border-radius: var(--radius); padding: 20px; margin-bottom: 12px; cursor: pointer; transition: var(--transition); position: relative; box-shadow: var(--shadow-sm); }
  .plan-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
  .plan-card.featured { border-color: var(--yellow); background: linear-gradient(180deg, #FFFEF0 0%, var(--bg-card) 100%); }
  .plan-card.selected { border-color: var(--green); background: #F0F8F2; }
  .plan-badge-featured { position: absolute; top: -10px; left: 50%; transform: translateX(-50%); background: var(--yellow); color: var(--green); font-size: 11px; font-weight: 700; padding: 4px 14px; border-radius: 20px; white-space: nowrap; }
  .plan-name { font-size: 18px; font-weight: 700; color: var(--text); margin-bottom: 4px; }
  .plan-price { font-family: 'Space Grotesk', sans-serif; font-size: 36px; font-weight: 700; color: var(--green); margin-bottom: 12px; }
  .plan-price span { font-size: 16px; color: var(--text-muted); }
  .plan-features { list-style: none; }
  .plan-features li { font-size: 13px; color: var(--text-soft); padding: 4px 0; display: flex; align-items: center; gap: 8px; }
  .plan-features li::before { content: '✓'; color: var(--green); font-weight: 700; }

  /* PAYMENT */
  .payment-card { background: var(--bg-card); border: 1.5px solid var(--border); border-radius: var(--radius); padding: 18px; margin-bottom: 10px; display: flex; align-items: center; gap: 16px; cursor: pointer; transition: var(--transition); box-shadow: var(--shadow-sm); }
  .payment-card:hover { border-color: var(--green); transform: translateY(-1px); }
  .payment-icon { width: 48px; height: 48px; background: var(--yellow); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; flex-shrink: 0; }
  .payment-info { flex: 1; }
  .payment-name { font-size: 15px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
  .payment-desc { font-size: 12px; color: var(--text-muted); }

  /* GRANTED */
  .granted-screen { display: flex; flex-direction: column; align-items: center; justify-content: center; flex: 1; padding: 40px 28px; text-align: center; gap: 16px; }
  .granted-icon { width: 100px; height: 100px; background: var(--yellow); border: 3px solid var(--green); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 48px; margin-bottom: 16px; animation: pulse 2s infinite; color: var(--green); }
  .granted-title { font-family: 'Space Grotesk', sans-serif; font-size: 28px; font-weight: 700; color: var(--green); }
  .granted-sub { font-size: 15px; color: var(--text-soft); line-height: 1.6; }

  /* DENIED */
  .denied-screen { display: flex; flex-direction: column; align-items: center; justify-content: center; flex: 1; padding: 40px 28px; text-align: center; gap: 16px; }
  .denied-icon { width: 100px; height: 100px; background: rgba(224,72,72,0.1); border: 3px solid var(--red); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 48px; margin-bottom: 16px; color: var(--red); }
  .denied-title { font-family: 'Space Grotesk', sans-serif; font-size: 28px; font-weight: 700; color: var(--red); }
  .denied-sub { font-size: 15px; color: var(--text-soft); line-height: 1.6; }

  /* PROCESSING */
  .processing-screen { display: flex; flex-direction: column; align-items: center; justify-content: center; flex: 1; padding: 40px; gap: 24px; text-align: center; }
  .spinner { width: 80px; height: 80px; border: 4px solid var(--border); border-top-color: var(--green); border-radius: 50%; animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* HOME */
  #screen-home { padding-bottom: 120px; }
  .home-header { padding: 32px 24px 28px; background: linear-gradient(180deg, #FFFFFF 0%, var(--bg) 100%); position: relative; overflow: hidden; margin-bottom: 8px; border-bottom: 1px solid var(--border); }
  .home-header::after { content: ''; position: absolute; bottom: -30px; right: -20px; width: 120px; height: 120px; background: var(--yellow); opacity: 0.3; border-radius: 50%; }
  .home-greeting { color: var(--text-muted); font-size: 14px; margin-bottom: 4px; position: relative; z-index: 1; }
  .home-name { font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; color: var(--text); margin-bottom: 20px; position: relative; z-index: 1; }
  .home-name span { color: var(--green); }
  .home-card { background: var(--green); border-radius: 20px; padding: 22px; display: flex; justify-content: space-between; align-items: center; box-shadow: var(--shadow-md); position: relative; z-index: 1; overflow: hidden; }
  .home-card::before { content: ''; position: absolute; top: -20px; right: -20px; width: 80px; height: 80px; background: var(--yellow); border-radius: 50%; opacity: 0.15; }
  .home-card-label { font-size: 12px; font-weight: 600; color: var(--yellow); opacity: 0.9; text-transform: uppercase; letter-spacing: 0.5px; }
  .home-card-value { font-family: 'Space Grotesk', sans-serif; font-size: 32px; font-weight: 700; color: white; }
  .home-card-sub { font-size: 13px; color: rgba(255,255,255,0.7); }
  .home-card-icon { font-size: 40px; position: relative; z-index: 1; }

  .menu-section-label {
    font-size: 11px; font-weight: 700; color: var(--text-muted); letter-spacing: 1px; text-transform: uppercase;
    padding: 24px 24px 12px; display: flex; align-items: center; gap: 8px;
  }
  .menu-section-label::before { content: ''; width: 14px; height: 2px; background: var(--green); }

  .menu-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 16px 24px 0; }
  .menu-card { background: var(--bg-card); border: 1.5px solid var(--border); border-radius: var(--radius); padding: 20px; cursor: pointer; transition: var(--transition); position: relative; overflow: hidden; box-shadow: var(--shadow-sm); }
  .menu-card:hover { border-color: var(--green); transform: translateY(-2px); box-shadow: var(--shadow-md); }
  .menu-card-icon { font-size: 28px; margin-bottom: 10px; }
  .menu-card-title { font-size: 14px; font-weight: 700; color: var(--text); margin-bottom: 4px; }
  .menu-card-sub { font-size: 12px; color: var(--text-muted); }
  .menu-card-arrow { position: absolute; bottom: 16px; right: 16px; width: 24px; height: 24px; background: var(--yellow); border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 12px; color: var(--green); }

  /* UPGRADE BANNER */
  .upgrade-banner { margin: 16px 24px 0; background: linear-gradient(135deg, var(--green) 0%, var(--green-mid) 100%); border-radius: var(--radius); padding: 18px; display: flex; align-items: center; gap: 14px; cursor: pointer; transition: var(--transition); position: relative; overflow: hidden; box-shadow: var(--shadow-md); }
  .upgrade-banner::before { content: ''; position: absolute; top: -40px; right: -40px; width: 120px; height: 120px; background: var(--yellow); border-radius: 50%; opacity: 0.15; }
  .upgrade-banner:hover { transform: translateY(-2px); }
  .upgrade-icon { width: 44px; height: 44px; background: var(--yellow); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; flex-shrink: 0; position: relative; z-index: 1; }
  .upgrade-text { flex: 1; position: relative; z-index: 1; }
  .upgrade-title { font-size: 14px; font-weight: 700; color: white; margin-bottom: 2px; }
  .upgrade-sub { font-size: 12px; color: rgba(255,255,255,0.7); }
  .upgrade-arrow { color: var(--yellow); font-size: 18px; position: relative; z-index: 1; }

  /* BOTTOM NAV */
  .bottom-nav { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(255,255,255,0.96); backdrop-filter: blur(20px); border-top: 1px solid var(--border); display: flex; padding: 12px 0 24px; }
  .nav-item { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px; cursor: pointer; transition: var(--transition); }
  .nav-icon { font-size: 22px; }
  .nav-label { font-size: 10px; font-weight: 600; color: var(--text-muted); letter-spacing: 0.3px; }
  .nav-item.active .nav-label { color: var(--green); }
  .nav-item.active .nav-icon { transform: scale(1.1); }

  /* NUTRICION */
  .objective-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; margin-bottom: 20px; }
  .objective-card { background: var(--bg-card); border: 1.5px solid var(--border); border-radius: var(--radius-sm); padding: 16px; cursor: pointer; transition: var(--transition); text-align: center; box-shadow: var(--shadow-sm); }
  .objective-card:hover, .objective-card.selected { border-color: var(--green); background: #F0F8F2; }
  .objective-card .obj-icon { font-size: 28px; margin-bottom: 8px; }
  .objective-card .obj-label { font-size: 13px; font-weight: 600; color: var(--text); }
  .diet-card { background: var(--bg-card); border: 1.5px solid var(--border); border-radius: var(--radius); padding: 20px; margin-bottom: 12px; cursor: pointer; transition: var(--transition); box-shadow: var(--shadow-sm); }
  .diet-card:hover { border-color: var(--green); transform: translateY(-2px); }
  .diet-card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
  .diet-name { font-size: 16px; font-weight: 700; color: var(--text); }
  .diet-cal { background: var(--yellow); color: var(--green); font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 20px; }
  .diet-desc { font-size: 13px; color: var(--text-soft); line-height: 1.5; margin-bottom: 12px; }
  .diet-macros { display: flex; gap: 12px; padding-top: 12px; border-top: 1px solid var(--border); }
  .macro { text-align: center; }
  .macro-val { font-size: 16px; font-weight: 700; color: var(--green); }
  .macro-label { font-size: 10px; color: var(--text-muted); }

  /* DIETA DETALLADA */
  .meal-row { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 14px; margin-bottom: 10px; }
  .meal-time { font-size: 13px; color: var(--green); font-weight: 700; margin-bottom: 6px; }
  .meal-desc { font-size: 12px; color: var(--text-soft); line-height: 1.5; }

  /* VIDA PUNTOS */
  .points-hero { margin: 0 24px 20px; background: var(--green); border-radius: var(--radius); padding: 24px; text-align: center; position: relative; overflow: hidden; box-shadow: var(--shadow-md); }
  .points-hero::before { content: ''; position: absolute; top: -50px; right: -50px; width: 160px; height: 160px; background: var(--yellow); border-radius: 50%; opacity: 0.1; }
  .points-label { color: var(--yellow); font-size: 13px; margin-bottom: 8px; position: relative; z-index: 1; }
  .points-value { font-family: 'Space Grotesk', sans-serif; font-size: 52px; font-weight: 700; color: white; line-height: 1; position: relative; z-index: 1; }
  .points-sub { color: rgba(255,255,255,0.7); font-size: 13px; margin-top: 8px; position: relative; z-index: 1; }
  .points-saldo-corner { position: absolute; top: 14px; right: 24px; background: var(--green); color: var(--yellow); font-size: 12px; font-weight: 700; padding: 6px 12px; border-radius: 20px; z-index: 10; }
  .benefit-card { background: var(--bg-card); border: 1.5px solid var(--border); border-radius: var(--radius); padding: 16px; margin-bottom: 10px; display: flex; align-items: center; gap: 16px; cursor: pointer; transition: var(--transition); box-shadow: var(--shadow-sm); }
  .benefit-card:hover { border-color: var(--green); transform: translateY(-1px); }
  .benefit-icon-wrap { width: 48px; height: 48px; background: var(--yellow); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; flex-shrink: 0; }
  .benefit-info { flex: 1; }
  .benefit-name { font-size: 15px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
  .benefit-desc { font-size: 12px; color: var(--text-muted); }
  .benefit-cost { font-size: 14px; font-weight: 700; color: var(--green); }

  .obtencion-list { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; }
  .obtencion-item { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid var(--border); }
  .obtencion-item:last-child { border-bottom: none; }
  .obtencion-label { font-size: 14px; color: var(--text); }
  .obtencion-value { font-size: 14px; font-weight: 700; color: var(--green); background: var(--yellow); padding: 4px 10px; border-radius: 20px; }

  /* CLASES */
  .filter-row { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 4px; margin-bottom: 16px; scrollbar-width: none; }
  .filter-row::-webkit-scrollbar { display: none; }
  .filter-chip { padding: 8px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; white-space: nowrap; cursor: pointer; transition: var(--transition); border: 1.5px solid var(--border); background: var(--bg-soft); color: var(--text-soft); font-family: 'DM Sans', sans-serif; }
  .filter-chip.active { background: var(--green); color: white; border-color: var(--green); }
  .class-card { background: var(--bg-card); border: 1.5px solid var(--border); border-radius: var(--radius); overflow: hidden; margin-bottom: 12px; cursor: pointer; transition: var(--transition); box-shadow: var(--shadow-sm); }
  .class-card:hover { border-color: var(--green); transform: translateY(-2px); }
  .class-thumb { height: 90px; display: flex; align-items: center; justify-content: center; font-size: 40px; position: relative; }
  .class-badge { position: absolute; top: 10px; right: 10px; background: var(--yellow); color: var(--green); font-size: 10px; font-weight: 700; padding: 4px 8px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.5px; }
  .class-tipo-badge { position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); color: var(--text); font-size: 10px; font-weight: 600; padding: 4px 8px; border-radius: 20px; }
  .class-body { padding: 14px; background: var(--bg-card); }
  .class-name { font-size: 15px; font-weight: 700; color: var(--text); margin-bottom: 6px; }
  .class-meta { display: flex; gap: 12px; flex-wrap: wrap; }
  .class-meta-item { font-size: 12px; color: var(--text-muted); display: flex; align-items: center; gap: 4px; }

  /* HORARIO PICKER */
  .horario-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-bottom: 20px; }
  .horario-chip { padding: 12px; text-align: center; border-radius: var(--radius-sm); background: var(--bg-card); border: 1.5px solid var(--border); font-size: 13px; font-weight: 600; color: var(--text); cursor: pointer; transition: var(--transition); }
  .horario-chip:hover { border-color: var(--green); }
  .horario-chip.selected { background: var(--green); color: white; border-color: var(--green); }
  .horario-chip.unavailable { opacity: 0.3; cursor: not-allowed; }

  /* CLASS DETAIL */
  .class-detail-hero { height: 160px; background: var(--green); display: flex; align-items: center; justify-content: center; font-size: 64px; margin: 0 -24px 20px; position: relative; overflow: hidden; }
  .class-detail-hero::before { content: ''; position: absolute; top: -40px; right: -40px; width: 160px; height: 160px; background: var(--yellow); border-radius: 50%; opacity: 0.15; }
  .class-detail-meta { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
  .class-detail-meta span { background: var(--bg-soft); color: var(--text); font-size: 12px; padding: 5px 12px; border-radius: 20px; border: 1px solid var(--border); }
  .class-detail-meta span.yellow { background: var(--yellow); color: var(--green); font-weight: 700; border-color: var(--yellow); }

  /* QR */
  .qr-box { width: 200px; height: 200px; background: var(--bg-card); border-radius: 20px; padding: 16px; display: flex; align-items: center; justify-content: center; box-shadow: var(--shadow-lg); margin: 0 auto 20px; border: 2px solid var(--green); }
  .qr-inner { width: 168px; height: 168px; background: repeating-conic-gradient(var(--green) 0% 25%, var(--bg-card) 0% 50%) 0 0/16px 16px; border-radius: 8px; }
  .qr-timer { background: var(--green); color: var(--yellow); border-radius: 30px; padding: 10px 24px; font-size: 14px; font-weight: 600; display: flex; align-items: center; gap: 8px; margin: 0 auto 12px; width: fit-content; }
  .qr-timer-dot { width: 8px; height: 8px; background: var(--yellow); border-radius: 50%; animation: blink 1s infinite; }
  @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0; } }
  .qr-valid-badge { background: var(--yellow); color: var(--green); font-size: 12px; font-weight: 700; padding: 6px 14px; border-radius: 20px; display: inline-block; margin: 12px auto 0; }
  .bio-icon-big { width: 120px; height: 120px; background: var(--yellow); border: 3px solid var(--green); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 56px; margin: 0 auto 24px; animation: pulse 2s infinite; }
  .face-scan-frame { width: 220px; height: 220px; margin: 0 auto 20px; position: relative; border-radius: 24px; overflow: hidden; background: var(--bg-card); border: 2px solid var(--border); display: flex; align-items: center; justify-content: center; box-shadow: var(--shadow-md); }
  .face-scan-frame .corner { position: absolute; width: 28px; height: 28px; border-color: var(--green); border-style: solid; }
  .face-scan-frame .c-tl { top: 10px; left: 10px; border-width: 3px 0 0 3px; border-top-left-radius: 8px; }
  .face-scan-frame .c-tr { top: 10px; right: 10px; border-width: 3px 3px 0 0; border-top-right-radius: 8px; }
  .face-scan-frame .c-bl { bottom: 10px; left: 10px; border-width: 0 0 3px 3px; border-bottom-left-radius: 8px; }
  .face-scan-frame .c-br { bottom: 10px; right: 10px; border-width: 0 3px 3px 0; border-bottom-right-radius: 8px; }
  .face-scan-frame .face-emoji { font-size: 100px; }
  .face-scan-line { position: absolute; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, transparent, var(--yellow), transparent); animation: scanline 2s linear infinite; }
  @keyframes scanline { 0% { top: 10%; } 50% { top: 90%; } 100% { top: 10%; } }

  /* SEDES */
  .sede-card { background: var(--bg-card); border: 1.5px solid var(--border); border-radius: var(--radius); padding: 18px; margin-bottom: 10px; cursor: pointer; transition: var(--transition); box-shadow: var(--shadow-sm); }
  .sede-card:hover { border-color: var(--green); transform: translateY(-1px); }
  .sede-card-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }
  .sede-name { font-size: 16px; font-weight: 700; color: var(--text); }
  .sede-open { font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; }
  .sede-open.open { background: var(--yellow); color: var(--green); }
  .sede-open.closed { background: rgba(224,72,72,0.1); color: var(--red); }
  .sede-address { font-size: 13px; color: var(--text-muted); margin-bottom: 12px; }
  .sede-info-row { display: flex; gap: 16px; flex-wrap: wrap; }
  .sede-info-item { font-size: 12px; color: var(--text-soft); display: flex; align-items: center; gap: 4px; }

  .map-placeholder { background: linear-gradient(135deg, #E8F0E5, #F0F5EC); border-radius: var(--radius); height: 140px; display: flex; align-items: center; justify-content: center; margin-bottom: 20px; border: 1.5px solid var(--border); flex-direction: column; gap: 8px; }

  /* SEDE DETAIL */
  .detail-option { background: var(--bg-card); border: 1.5px solid var(--border); border-radius: var(--radius); padding: 18px; margin-bottom: 10px; cursor: pointer; transition: var(--transition); display: flex; align-items: center; gap: 14px; box-shadow: var(--shadow-sm); }
  .detail-option:hover { border-color: var(--green); }
  .detail-option-icon { width: 44px; height: 44px; background: var(--yellow); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink: 0; }
  .detail-option-text { flex: 1; }
  .detail-option-title { font-size: 15px; font-weight: 700; color: var(--text); margin-bottom: 2px; }
  .detail-option-sub { font-size: 12px; color: var(--text-muted); }

  .info-box { background: var(--bg-card); border-radius: var(--radius); padding: 18px; margin-bottom: 12px; border: 1px solid var(--border); box-shadow: var(--shadow-sm); }
  .info-box-title { font-size: 12px; font-weight: 700; color: var(--text-muted); letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px; }
  .info-box-content { font-size: 14px; color: var(--text); line-height: 1.7; }

  /* RUTINAS */
  .routine-card { background: var(--bg-card); border: 1.5px solid var(--border); border-radius: var(--radius); padding: 18px; margin-bottom: 10px; cursor: pointer; transition: var(--transition); box-shadow: var(--shadow-sm); }
  .routine-card:hover { border-color: var(--green); }
  .routine-name { font-size: 16px; font-weight: 700; color: var(--text); margin-bottom: 6px; }
  .routine-obj { display: inline-block; background: var(--yellow); color: var(--green); font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; margin-bottom: 10px; }
  .routine-exercises { font-size: 13px; color: var(--text-soft); }

  /* ALERT */
  .alert { display: none; position: absolute; top: 60px; left: 20px; right: 20px; padding: 14px 18px; border-radius: var(--radius-sm); font-size: 14px; font-weight: 600; z-index: 200; animation: slideDown 0.3s ease; box-shadow: var(--shadow-lg); }
  .alert.show { display: flex; align-items: center; gap: 10px; }
  .alert.success { background: var(--green); color: var(--yellow); }
  .alert.error { background: var(--red); color: white; }
  @keyframes slideDown { from { transform: translateY(-20px); opacity:0; } to { transform: translateY(0); opacity:1; } }

  .text-green { color: var(--green); font-weight: 700; }
  .gap-info { font-size: 13px; color: var(--text-muted); text-align: center; margin-top: 12px; }

  /* YOUTUBE COMPACT LINKS */
  .yt-section-label { font-size: 11px; font-weight: 700; color: var(--text-muted); letter-spacing: 1px; text-transform: uppercase; padding: 16px 24px 10px; }
  .yt-link { display: flex; align-items: center; gap: 12px; padding: 12px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 8px; text-decoration: none; transition: var(--transition); box-shadow: var(--shadow-sm); }
  .yt-link:hover { background: var(--bg-elevated); border-color: var(--green); }
  .yt-link-icon { width: 32px; height: 32px; background: #FF0000; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; flex-shrink: 0; }
  .yt-link-text { flex: 1; min-width: 0; }
  .yt-link-title { font-size: 13px; font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .yt-link-meta { font-size: 11px; color: var(--text-muted); margin-top: 2px; }

  /* PERFIL / SUSCRIPCION */
  .perfil-hero { background: var(--green); border-radius: var(--radius); padding: 20px; margin-bottom: 16px; position: relative; overflow: hidden; box-shadow: var(--shadow-md); }
  .perfil-hero::before { content: ''; position: absolute; top: -30px; right: -30px; width: 120px; height: 120px; background: var(--yellow); border-radius: 50%; opacity: 0.12; }
  .perfil-avatar { width: 64px; height: 64px; background: var(--yellow); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 28px; margin-bottom: 12px; position: relative; z-index: 1; }
  .perfil-name { font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; color: white; margin-bottom: 4px; position: relative; z-index: 1; }
  .perfil-mail { font-size: 13px; color: rgba(255,255,255,0.7); position: relative; z-index: 1; }
  .perfil-badge { display: inline-block; background: var(--yellow); color: var(--green); font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 20px; margin-top: 10px; position: relative; z-index: 1; }

  .config-row { background: var(--bg-card); border: 1.5px solid var(--border); border-radius: var(--radius-sm); padding: 16px; margin-bottom: 8px; display: flex; align-items: center; gap: 14px; cursor: pointer; transition: var(--transition); box-shadow: var(--shadow-sm); }
  .config-row:hover { border-color: var(--green); }
  .config-row-icon { width: 40px; height: 40px; background: var(--yellow); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; }
  .config-row-text { flex: 1; }
  .config-row-title { font-size: 14px; font-weight: 600; color: var(--text); }
  .config-row-sub { font-size: 11px; color: var(--text-muted); margin-top: 2px; }

  /* TARJETA FORM */
  .card-preview { background: var(--green); border-radius: var(--radius); padding: 24px; margin-bottom: 20px; position: relative; overflow: hidden; box-shadow: var(--shadow-md); }
  .card-preview::after { content: ''; position: absolute; top: -30px; right: -30px; width: 120px; height: 120px; background: var(--yellow); opacity: 0.15; border-radius: 50%; }
  .card-chip { width: 36px; height: 28px; background: linear-gradient(135deg, #D4AF37, #F5D76E); border-radius: 6px; margin-bottom: 16px; position: relative; z-index: 1; }
  .card-number-display { font-family: 'Space Grotesk', sans-serif; font-size: 18px; font-weight: 600; color: white; letter-spacing: 2px; margin-bottom: 16px; position: relative; z-index: 1; }
  .card-footer { display: flex; justify-content: space-between; position: relative; z-index: 1; }
  .card-footer-label { font-size: 9px; color: rgba(255,255,255,0.6); text-transform: uppercase; letter-spacing: 1px; }
  .card-footer-value { font-size: 13px; color: white; font-weight: 600; margin-top: 2px; }
  .card-brand { position: absolute; top: 20px; right: 20px; font-size: 22px; font-weight: 700; color: var(--yellow); font-family: 'Space Grotesk', sans-serif; z-index: 1; }
  .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
</style>
</head>
<body>
<div class="phone-frame">

  <div class="alert" id="global-alert"></div>

  <!-- SPLASH -->
  <div class="screen active" id="screen-splash">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div style="flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:24px; padding:40px;">
      <div class="splash-logo">🏋️</div>
      <div class="splash-title">Fit<span>Life</span></div>
      <div class="splash-sub">Tu gimnasio, en tu bolsillo</div>
      <div style="width:100%; margin-top:20px;">
        <button class="btn-primary" onclick="goto('auth-choice')">Ingresar a la app</button>
      </div>
    </div>
  </div>

  <!-- AUTH CHOICE -->
  <div class="screen" id="screen-auth-choice">
    <div class="auth-hero">
      <div class="status-bar" style="padding:0 0 16px;"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
      <div class="auth-hero-tag">Bienvenido</div>
      <h1>Entrena más<br><em>inteligente</em></h1>
      <p>Tu app de gimnasio todo en uno</p>
    </div>
    <div style="padding: 32px 28px; flex: 1;">
      <div class="step-title" style="font-size:20px;">¿Iniciar sesión o Registrarse?</div>
      <div class="step-sub">Elegí cómo querés ingresar</div>

      <div class="choice-card" onclick="goto('login')">
        <div class="choice-card-icon">🔑</div>
        <div class="choice-card-info">
          <div class="choice-card-title">Iniciar sesión</div>
          <div class="choice-card-sub">Ya tengo una cuenta</div>
        </div>
        <div class="choice-card-arrow">→</div>
      </div>

      <div class="choice-card" onclick="goto('register-mail')">
        <div class="choice-card-icon">✨</div>
        <div class="choice-card-info">
          <div class="choice-card-title">Registrarse</div>
          <div class="choice-card-sub">Crear cuenta nueva</div>
        </div>
        <div class="choice-card-arrow">→</div>
      </div>
    </div>
  </div>

  <!-- LOGIN -->
  <div class="screen" id="screen-login">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail">
      <span>Iniciar sesión</span><span class="sep">›</span><span class="current">Ingresar credenciales</span>
    </div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('auth-choice')">←</button>
      <div class="top-nav-title">Iniciar sesión</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Ingresar credenciales</div>
      <div class="step-sub">Usá el email y contraseña con los que te registraste</div>

      <div class="form-group">
        <label class="form-label">Email</label>
        <input class="form-input" type="email" placeholder="tucorreo@gmail.com" id="login-email" value="usuario@fitlife.com">
      </div>
      <div class="form-group">
        <label class="form-label">Contraseña</label>
        <input class="form-input" type="password" placeholder="••••••••" id="login-pass" value="demo1234">
      </div>

      <button class="btn-primary" onclick="doLogin()">Iniciar sesión</button>
      <p class="gap-info">Tip: usar <span class="text-green">vencida@test.com</span> simula cuota vencida</p>
    </div>
  </div>

  <!-- CUOTA CHECK -->
  <div class="screen" id="screen-cuota-check">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Login</span><span class="sep">›</span><span class="current">Verificando estado de cuota</span></div>
    <div class="decision-screen">
      <div class="decision-diamond">
        <div class="decision-diamond-inner">
          <div class="q-icon">💰</div>
          <div class="q-text">¿Tiene cuota<br>al día?</div>
        </div>
      </div>
      <div class="decision-context">Verificamos el estado de tu suscripción</div>
      <div style="width:100%;" id="cuota-result">
        <div class="spinner" style="margin: 0 auto;"></div>
      </div>
    </div>
  </div>

  <!-- RENOVAR -->
  <div class="screen" id="screen-renovar-check">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Login</span><span class="sep">›</span><span>Cuota vencida</span><span class="sep">›</span><span class="current">¿Renovar?</span></div>
    <div class="decision-screen">
      <div class="decision-diamond">
        <div class="decision-diamond-inner">
          <div class="q-icon">🔄</div>
          <div class="q-text">¿Renovar<br>suscripción?</div>
        </div>
      </div>
      <div class="decision-context">Tu membresía venció. Renová tu plan para seguir accediendo al contenido</div>
      <div class="decision-actions">
        <button class="btn-danger" onclick="goto('access-denied')" style="margin-top:0;">No</button>
        <button class="btn-primary" onclick="goto('payment-method')">Sí, renovar</button>
      </div>
    </div>
  </div>

  <!-- REGISTER -->
  <div class="screen" id="screen-register-mail">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail">
      <span>Registro</span><span class="sep">›</span><span class="current">Mail y contraseña</span><span class="sep">›</span><span>Confirmar mail</span><span class="sep">›</span><span>T&C</span><span class="sep">›</span><span>Plan</span><span class="sep">›</span><span>Pago</span>
    </div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('auth-choice')">←</button>
      <div class="top-nav-title">Crear cuenta</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Ingresar mail y contraseña</div>
      <div class="step-sub">Empezamos con tus datos de acceso</div>

      <div class="form-group">
        <label class="form-label">Nombre completo</label>
        <input class="form-input" type="text" placeholder="Juan García" id="reg-name" value="Santiago Estevez">
      </div>
      <div class="form-group">
        <label class="form-label">Email</label>
        <input class="form-input" type="email" placeholder="tucorreo@gmail.com" id="reg-email" value="santiago@fitlife.com">
      </div>
      <div class="form-group">
        <label class="form-label">Contraseña</label>
        <input class="form-input" type="password" placeholder="••••••••" id="reg-pass" value="demo1234">
      </div>

      <button class="btn-primary" onclick="doRegister()">Continuar →</button>
    </div>
  </div>

  <!-- SEND MAIL -->
  <div class="screen" id="screen-send-mail">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail">
      <span>Registro</span><span class="sep">›</span><span>Mail</span><span class="sep">›</span><span class="current">Enviando confirmación</span>
    </div>
    <div class="decision-screen">
      <div class="bio-icon-big">📨</div>
      <div class="step-title" style="text-align:center;">Enviar mail de confirmación</div>
      <div class="decision-context">Te estamos enviando un correo a:</div>
      <div class="mail-addr" id="sent-mail-addr" style="font-size:16px; text-align:center;">santiago@fitlife.com</div>
      <div class="spinner" style="margin: 8px auto;"></div>
    </div>
  </div>

  <!-- CONFIRM MAIL -->
  <div class="screen" id="screen-confirm-mail">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail">
      <span>Registro</span><span class="sep">›</span><span>Mail</span><span class="sep">›</span><span class="current">Confirmar mail</span>
    </div>
    <div class="top-nav">
      <div class="top-nav-title">Confirmar mail</div>
    </div>
    <div class="screen-content">
      <div class="step-title">Confirmar mail</div>
      <div class="step-sub">Revisá tu bandeja de entrada y confirmá el correo</div>

      <div class="mail-card">
        <div class="mail-icon">📧</div>
        <div class="mail-status">Mail enviado a</div>
        <div class="mail-addr" id="confirm-mail-addr">santiago@fitlife.com</div>
      </div>

      <div style="background:var(--bg-soft); border:1px dashed var(--border-strong); border-radius:10px; padding:14px; margin-bottom:24px;">
        <div style="font-size:12px; color:var(--text-muted); margin-bottom:6px;">📥 Asunto</div>
        <div style="font-size:13px; color:var(--text); margin-bottom:10px;">Confirmá tu cuenta FitLife</div>
        <div style="font-size:12px; color:var(--text-muted);">Hacé click en el enlace del mail para verificar tu cuenta</div>
      </div>

      <button class="btn-primary" onclick="goto('terms')">Ya confirmé mi mail ✓</button>
      <button class="btn-secondary" onclick="showAlert('Mail reenviado', 'success')">Reenviar mail</button>
    </div>
  </div>

  <!-- TERMS -->
  <div class="screen" id="screen-terms">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail">
      <span>Registro</span><span class="sep">›</span><span>Mail</span><span class="sep">›</span><span class="current">Firmar T&C</span>
    </div>
    <div class="top-nav">
      <div class="top-nav-title">Términos y condiciones</div>
    </div>
    <div class="screen-content">
      <div class="step-title">Firmar términos y condiciones</div>
      <div class="step-sub">Leé y aceptá los términos para continuar</div>

      <div class="terms-box">
        <h4>1. Política de privacidad</h4>
        <p>Tus datos biométricos, de salud y personales son confidenciales. No se comparten con terceros sin tu consentimiento.</p>
        <h4>2. Condiciones de membresía</h4>
        <p>La membresía se renueva automáticamente cada mes. Podés cancelar en cualquier momento desde tu perfil.</p>
        <h4>3. Uso responsable</h4>
        <p>Queda prohibido el mal uso de las instalaciones, la reventa de beneficios y el préstamo del QR de acceso.</p>
        <h4>4. Política de cancelación</h4>
        <p>Las cancelaciones se procesan en hasta 5 días hábiles. Los reintegros se realizan al medio de pago original.</p>
        <h4>5. Sistema Vida Puntos</h4>
        <p>Los puntos no son canjeables por dinero ni transferibles. Vencen al cabo de 12 meses sin uso.</p>
      </div>

      <div class="terms-checkbox">
        <input type="checkbox" id="terms-accept">
        <label for="terms-accept">He leído y acepto los términos y condiciones</label>
      </div>

      <button class="btn-primary" onclick="doSignTerms()">Firmar y continuar</button>
      <button class="btn-secondary" onclick="goto('access-denied')">No acepto</button>
    </div>
  </div>

  <!-- PLAN -->
  <div class="screen" id="screen-plan">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail">
      <span>Suscripción</span><span class="sep">›</span><span class="current">Elegir plan</span><span class="sep">›</span><span>Pago</span>
    </div>
    <div class="top-nav">
      <button class="back-btn" onclick="_registering ? goto('terms') : goto('perfil')">←</button>
      <div class="top-nav-title">Elegir plan</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Elegir plan de suscripción</div>
      <div class="step-sub">Seleccioná el plan que mejor se adapte</div>

      <div class="plan-card" onclick="selectPlan(this, 'Básico', '$8.999')">
        <div class="plan-name">Básico</div>
        <div class="plan-price">$8.999 <span>/mes</span></div>
        <ul class="plan-features">
          <li>Acceso a 1 sede</li>
          <li>Clases virtuales ilimitadas</li>
          <li>Plan de rutinas básico</li>
        </ul>
      </div>

      <div class="plan-card featured" onclick="selectPlan(this, 'Premium', '$14.999')">
        <div class="plan-badge-featured">⭐ Más popular</div>
        <div class="plan-name">Premium</div>
        <div class="plan-price">$14.999 <span>/mes</span></div>
        <ul class="plan-features">
          <li>Acceso a todas las sedes</li>
          <li>Plan nutricional personalizado</li>
          <li>Sistema Vida Puntos</li>
          <li>Reserva de máquinas</li>
        </ul>
      </div>

      <div class="plan-card" onclick="selectPlan(this, 'Elite', '$24.999')">
        <div class="plan-name">Elite</div>
        <div class="plan-price">$24.999 <span>/mes</span></div>
        <ul class="plan-features">
          <li>Todo lo del plan Premium</li>
          <li>Entrenador personal virtual</li>
          <li>Doble de puntos Vida</li>
        </ul>
      </div>

      <button class="btn-primary" id="btn-continue-plan" onclick="goto('pagar-check')" style="margin-top:20px; opacity:0.5; pointer-events:none;">Continuar →</button>
    </div>
  </div>

  <!-- PAGAR -->
  <div class="screen" id="screen-pagar-check">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail">
      <span>Plan</span><span class="sep">›</span><span class="current">¿Pagar?</span>
    </div>
    <div class="decision-screen">
      <div class="decision-diamond">
        <div class="decision-diamond-inner">
          <div class="q-icon">💳</div>
          <div class="q-text">¿Pagar?</div>
        </div>
      </div>
      <div class="decision-context">
        Confirmá que querés proceder al pago del plan <span class="text-green" id="pagar-plan-label">Premium</span>
      </div>
      <div class="decision-actions">
        <button class="btn-danger" onclick="_registering ? goto('access-denied') : goto('home')" style="margin-top:0;">No</button>
        <button class="btn-primary" onclick="goto('payment-method')">Sí, pagar</button>
      </div>
    </div>
  </div>

  <!-- PAYMENT METHOD -->
  <div class="screen" id="screen-payment-method">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail">
      <span>¿Pagar?</span><span class="sep">›</span><span class="current">Método de pago</span><span class="sep">›</span><span>Procesar</span>
    </div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('pagar-check')">←</button>
      <div class="top-nav-title">Método de pago</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Elegir método de pago</div>
      <div class="step-sub">Plan seleccionado: <span class="text-green" id="payment-plan-label">Premium — $14.999/mes</span></div>

      <div class="payment-card" onclick="elegirMetodo('Tarjeta de crédito')">
        <div class="payment-icon">💳</div>
        <div class="payment-info">
          <div class="payment-name">Tarjeta de crédito</div>
          <div class="payment-desc">Visa, Mastercard, Amex</div>
        </div>
        <span style="color:var(--green);">→</span>
      </div>

      <div class="payment-card" onclick="elegirMetodo('Tarjeta de débito')">
        <div class="payment-icon">💳</div>
        <div class="payment-info">
          <div class="payment-name">Tarjeta de débito</div>
          <div class="payment-desc">Débito en cuenta directo</div>
        </div>
        <span style="color:var(--green);">→</span>
      </div>

      <div class="payment-card" onclick="procesarPago('Transferencia bancaria')">
        <div class="payment-icon">🏦</div>
        <div class="payment-info">
          <div class="payment-name">Transferencia bancaria</div>
          <div class="payment-desc">CBU / CVU / Alias</div>
        </div>
        <span style="color:var(--green);">→</span>
      </div>

      <div class="payment-card" onclick="procesarPago('MercadoPago')">
        <div class="payment-icon">📱</div>
        <div class="payment-info">
          <div class="payment-name">MercadoPago</div>
          <div class="payment-desc">Cuenta MP o QR (simula fallo)</div>
        </div>
        <span style="color:var(--green);">→</span>
      </div>
    </div>
  </div>

  <!-- TARJETA FORM -->
  <div class="screen" id="screen-card-form">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Método</span><span class="sep">›</span><span class="current">Datos de tarjeta</span><span class="sep">›</span><span>Procesar</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('payment-method')">←</button>
      <div class="top-nav-title">Datos de tarjeta</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Ingresá los datos</div>
      <div class="step-sub" id="card-form-method">Tarjeta de crédito</div>

      <div class="card-preview">
        <div class="card-brand">VISA</div>
        <div class="card-chip"></div>
        <div class="card-number-display" id="prev-number">•••• •••• •••• ••••</div>
        <div class="card-footer">
          <div>
            <div class="card-footer-label">Titular</div>
            <div class="card-footer-value" id="prev-name">NOMBRE APELLIDO</div>
          </div>
          <div>
            <div class="card-footer-label">Vence</div>
            <div class="card-footer-value" id="prev-exp">MM/AA</div>
          </div>
        </div>
      </div>

      <div class="form-group">
        <label class="form-label">Número de tarjeta</label>
        <input class="form-input" type="text" inputmode="numeric" maxlength="19" placeholder="1234 5678 9012 3456" id="card-number" oninput="formatCardNumber(this)">
      </div>

      <div class="form-group">
        <label class="form-label">Nombre del titular</label>
        <input class="form-input" type="text" placeholder="Como figura en la tarjeta" id="card-name" oninput="updatePreview()">
      </div>

      <div class="form-row">
        <div class="form-group">
          <label class="form-label">Vencimiento</label>
          <input class="form-input" type="text" inputmode="numeric" maxlength="5" placeholder="MM/AA" id="card-exp" oninput="formatExp(this)">
        </div>
        <div class="form-group">
          <label class="form-label">CVV</label>
          <input class="form-input" type="text" inputmode="numeric" maxlength="4" placeholder="123" id="card-cvv">
        </div>
      </div>

      <button class="btn-primary" onclick="submitCard()">Pagar ahora</button>
    </div>
  </div>

  <!-- PERFIL / MI SUSCRIPCION -->
  <div class="screen" id="screen-perfil">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span class="current">Mi suscripción y cuenta</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('home')">←</button>
      <div class="top-nav-title">Mi Suscripción</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="perfil-hero">
        <div class="perfil-avatar">👤</div>
        <div class="perfil-name" id="perfil-name">Santiago Estevez</div>
        <div class="perfil-mail" id="perfil-mail">santiago@fitlife.com</div>
        <div class="perfil-badge" id="perfil-plan-badge">⭐ Plan Premium</div>
      </div>

      <div style="font-family:'Space Grotesk',sans-serif; font-size:14px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px; margin: 12px 0;">Suscripción</div>

      <div class="config-row" onclick="goto('plan')">
        <div class="config-row-icon">🔄</div>
        <div class="config-row-text">
          <div class="config-row-title">Cambiar plan</div>
          <div class="config-row-sub">Básico, Premium o Elite</div>
        </div>
        <span style="color:var(--green);">→</span>
      </div>

      <div class="config-row" onclick="goto('payment-method')">
        <div class="config-row-icon">💳</div>
        <div class="config-row-text">
          <div class="config-row-title">Método de pago</div>
          <div class="config-row-sub">Cambiar o actualizar</div>
        </div>
        <span style="color:var(--green);">→</span>
      </div>

      <div class="config-row" onclick="showAlert('Próxima en 12 días', 'success')">
        <div class="config-row-icon">📅</div>
        <div class="config-row-text">
          <div class="config-row-title">Próximo cobro</div>
          <div class="config-row-sub">15 de Junio · $14.999</div>
        </div>
        <span style="color:var(--green);">→</span>
      </div>

      <div class="config-row" onclick="showAlert('Historial cargado', 'success')">
        <div class="config-row-icon">📋</div>
        <div class="config-row-text">
          <div class="config-row-title">Historial de pagos</div>
          <div class="config-row-sub">Ver facturas anteriores</div>
        </div>
        <span style="color:var(--green);">→</span>
      </div>

      <div style="font-family:'Space Grotesk',sans-serif; font-size:14px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px; margin: 20px 0 12px;">Cuenta</div>

      <div class="config-row" onclick="showAlert('Datos personales', 'success')">
        <div class="config-row-icon">👤</div>
        <div class="config-row-text">
          <div class="config-row-title">Datos personales</div>
          <div class="config-row-sub">Nombre, email, teléfono</div>
        </div>
        <span style="color:var(--green);">→</span>
      </div>

      <div class="config-row" onclick="showAlert('Cambio de contraseña', 'success')">
        <div class="config-row-icon">🔒</div>
        <div class="config-row-text">
          <div class="config-row-title">Cambiar contraseña</div>
          <div class="config-row-sub">Seguridad de la cuenta</div>
        </div>
        <span style="color:var(--green);">→</span>
      </div>

      <div class="config-row" onclick="showAlert('Notificaciones', 'success')">
        <div class="config-row-icon">🔔</div>
        <div class="config-row-text">
          <div class="config-row-title">Notificaciones</div>
          <div class="config-row-sub">Push, email, recordatorios</div>
        </div>
        <span style="color:var(--green);">→</span>
      </div>

      <button class="btn-danger" onclick="if(confirm('¿Cancelar suscripción?')) goto('access-denied')">Cancelar suscripción</button>
      <button class="btn-secondary" onclick="goto('auth-choice')">Cerrar sesión</button>
    </div>
  </div>

  <!-- PROCESSING -->
  <div class="screen" id="screen-processing">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Método de pago</span><span class="sep">›</span><span class="current">Procesando</span></div>
    <div class="processing-screen">
      <div class="spinner"></div>
      <div class="step-title">Procesar pago</div>
      <div class="step-sub" style="margin-bottom:0;">Con <span class="text-green" id="processing-method-label">Tarjeta de crédito</span></div>
    </div>
  </div>

  <!-- PAGO EXITOSO -->
  <div class="screen" id="screen-pago-exitoso-check">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Procesar pago</span><span class="sep">›</span><span class="current">¿Exitoso?</span></div>
    <div class="decision-screen">
      <div class="decision-diamond">
        <div class="decision-diamond-inner">
          <div class="q-icon">✓</div>
          <div class="q-text">¿Pago<br>exitoso?</div>
        </div>
      </div>
      <div class="decision-context">Procesando respuesta del banco...</div>
      <div class="spinner" style="margin: 0 auto;"></div>
    </div>
  </div>

  <!-- REINTENTAR -->
  <div class="screen" id="screen-reintentar-check">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Pago fallido</span><span class="sep">›</span><span class="current">¿Reintentar?</span></div>
    <div class="decision-screen">
      <div class="decision-diamond" style="border-color:var(--red);">
        <div class="decision-diamond-inner">
          <div class="q-icon">⚠️</div>
          <div class="q-text">¿Reintentar?</div>
        </div>
      </div>
      <div class="decision-context">El pago no pudo procesarse. ¿Querés intentar de nuevo?</div>
      <div class="decision-actions">
        <button class="btn-danger" onclick="goto('home')" style="margin-top:0;">No</button>
        <button class="btn-primary" onclick="goto('payment-method')">Sí, reintentar</button>
      </div>
    </div>
  </div>

  <!-- ACCESS GRANTED -->
  <div class="screen" id="screen-access-granted">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="granted-screen">
      <div class="granted-icon">✓</div>
      <div class="granted-title">¡Bienvenido!</div>
      <div class="granted-sub">Tu cuenta está lista. Te redirigimos a la pantalla de inicio.</div>
      <div class="spinner" style="margin: 16px auto; width:40px; height:40px; border-width:3px;"></div>
    </div>
  </div>

  <!-- ACCESS DENIED -->
  <div class="screen" id="screen-access-denied">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="denied-screen">
      <div class="denied-icon">✕</div>
      <div class="denied-title">Acceso denegado</div>
      <div class="denied-sub">No podés acceder al contenido de la app sin una suscripción activa.</div>
      <div style="width:100%; margin-top:16px;">
        <button class="btn-primary" onclick="goto('auth-choice')">Volver al inicio</button>
      </div>
    </div>
  </div>

  <!-- HOME -->
  <div class="screen" id="screen-home">
    <div class="status-bar" style="padding:14px 28px 0; flex-shrink:0;"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="home-header">
      <div class="home-greeting">Buenos días 👋</div>
      <div class="home-name" id="home-username">Santiago <span>Estevez</span></div>
      <div class="home-card">
        <div>
          <div class="home-card-label">Vida Puntos</div>
          <div class="home-card-value" id="home-points">2.450</div>
          <div class="home-card-sub" id="home-plan-label">Plan Premium activo</div>
        </div>
        <div class="home-card-icon">⚡</div>
      </div>
    </div>

    <!-- BANNER UPGRADE -->
    <div class="upgrade-banner" onclick="goto('plan')" id="upgrade-banner-home">
      <div class="upgrade-icon">⭐</div>
      <div class="upgrade-text">
        <div class="upgrade-title">Suscribite o mejorá tu plan</div>
        <div class="upgrade-sub">Desbloqueá todos los beneficios</div>
      </div>
      <div class="upgrade-arrow">→</div>
    </div>

    <div class="menu-grid">
      <div class="menu-card" onclick="goto('nutrition')">
        <div class="menu-card-icon">🥗</div>
        <div class="menu-card-title">Plan de Nutrición</div>
        <div class="menu-card-sub">Recomendaciones</div>
        <div class="menu-card-arrow">→</div>
      </div>
      <div class="menu-card" onclick="goto('routines')">
        <div class="menu-card-icon">💪</div>
        <div class="menu-card-title">Rutinas</div>
        <div class="menu-card-sub">Personalizadas</div>
        <div class="menu-card-arrow">→</div>
      </div>
      <div class="menu-card" onclick="goto('qr-bio')">
        <div class="menu-card-icon">📱</div>
        <div class="menu-card-title">QR de acceso</div>
        <div class="menu-card-sub">Escaneo facial</div>
        <div class="menu-card-arrow">→</div>
      </div>
      <div class="menu-card" onclick="goto('perfil')">
        <div class="menu-card-icon">⚙️</div>
        <div class="menu-card-title">Mi Suscripción</div>
        <div class="menu-card-sub">Configurar cuenta</div>
        <div class="menu-card-arrow">→</div>
      </div>
    </div>

    <div class="yt-section-label">🎬 Entrenamientos destacados</div>
    <div style="padding: 0 24px 16px;">
      <a class="yt-link" href="https://www.youtube.com/watch?v=UBMk30rjy0o" target="_blank">
        <div class="yt-link-icon">▶</div>
        <div class="yt-link-text">
          <div class="yt-link-title">Full Body Workout — 30 min</div>
          <div class="yt-link-meta">🔥 Intermedio · 2.4M vistas</div>
        </div>
        <span style="color:var(--text-muted); font-size:14px;">↗</span>
      </a>
      <a class="yt-link" href="https://www.youtube.com/watch?v=ml6cT4AZdqI" target="_blank">
        <div class="yt-link-icon">▶</div>
        <div class="yt-link-text">
          <div class="yt-link-title">HIIT Cardio — 20 min sin equipo</div>
          <div class="yt-link-meta">⚡ Avanzado · 1.1M vistas</div>
        </div>
        <span style="color:var(--text-muted); font-size:14px;">↗</span>
      </a>
      <a class="yt-link" href="https://www.youtube.com/watch?v=v7AYKMP6rOE" target="_blank">
        <div class="yt-link-icon">▶</div>
        <div class="yt-link-text">
          <div class="yt-link-title">Yoga principiantes — 15 min</div>
          <div class="yt-link-meta">🧘 Principiante · 890K vistas</div>
        </div>
        <span style="color:var(--text-muted); font-size:14px;">↗</span>
      </a>
    </div>

    <div class="bottom-nav">
      <div class="nav-item active" onclick="goto('home')"><span class="nav-icon">🏠</span><span class="nav-label">Inicio</span></div>
      <div class="nav-item" onclick="goto('classes')"><span class="nav-icon">🎥</span><span class="nav-label">Clases</span></div>
      <div class="nav-item" onclick="goto('qr-bio')"><span class="nav-icon">📱</span><span class="nav-label">QR</span></div>
      <div class="nav-item" onclick="goto('sedes')"><span class="nav-icon">📍</span><span class="nav-label">Sedes</span></div>
      <div class="nav-item" onclick="goto('puntos-vida-check')"><span class="nav-icon">⚡</span><span class="nav-label">Puntos</span></div>
    </div>
  </div>

  <!-- NUTRICION -->
  <div class="screen" id="screen-nutrition">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span class="current">Ver recomendaciones nutricionales</span><span class="sep">›</span><span>Elegir objetivo</span><span class="sep">›</span><span>Dieta detallada</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('home')">←</button>
      <div class="top-nav-title">Plan de Nutrición</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Ver recomendaciones nutricionales</div>
      <div class="step-sub">Elegí en base a tus objetivos</div>

      <div class="objective-grid">
        <div class="objective-card" onclick="selectNutObj(this, 'deficit', '🔥 Perder peso')">
          <div class="obj-icon">🔥</div>
          <div class="obj-label">Perder peso</div>
        </div>
        <div class="objective-card" onclick="selectNutObj(this, 'mantenimiento', '⚖️ Mantenerme')">
          <div class="obj-icon">⚖️</div>
          <div class="obj-label">Mantenerme</div>
        </div>
        <div class="objective-card" onclick="selectNutObj(this, 'volumen', '💪 Ganar músculo')">
          <div class="obj-icon">💪</div>
          <div class="obj-label">Ganar músculo</div>
        </div>
        <div class="objective-card" onclick="selectNutObj(this, 'rendimiento', '⚡ Rendimiento')">
          <div class="obj-icon">⚡</div>
          <div class="obj-label">Rendimiento</div>
        </div>
      </div>

      <div style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:700; color:var(--text); margin: 8px 0 12px;">Dietas recomendadas</div>

      <div class="diet-card" onclick="goto('nutrition-detail', 'Mediterránea')">
        <div class="diet-card-header">
          <div class="diet-name">🫒 Mediterránea</div>
          <div class="diet-cal">1.800 kcal</div>
        </div>
        <div class="diet-desc">Vegetales, aceite de oliva, proteínas magras y cereales integrales.</div>
        <div class="diet-macros">
          <div class="macro"><div class="macro-val">45%</div><div class="macro-label">Carbos</div></div>
          <div class="macro"><div class="macro-val">30%</div><div class="macro-label">Proteína</div></div>
          <div class="macro"><div class="macro-val">25%</div><div class="macro-label">Grasas</div></div>
        </div>
      </div>

      <div class="diet-card" onclick="goto('nutrition-detail', 'Alta en proteínas')">
        <div class="diet-card-header">
          <div class="diet-name">🥩 Alta en proteínas</div>
          <div class="diet-cal">2.200 kcal</div>
        </div>
        <div class="diet-desc">Maximiza la síntesis muscular con alta ingesta proteica.</div>
        <div class="diet-macros">
          <div class="macro"><div class="macro-val">35%</div><div class="macro-label">Carbos</div></div>
          <div class="macro"><div class="macro-val">45%</div><div class="macro-label">Proteína</div></div>
          <div class="macro"><div class="macro-val">20%</div><div class="macro-label">Grasas</div></div>
        </div>
      </div>

      <div class="diet-card" onclick="goto('nutrition-detail', 'Plant-based')">
        <div class="diet-card-header">
          <div class="diet-name">🌱 Plant-based</div>
          <div class="diet-cal">1.600 kcal</div>
        </div>
        <div class="diet-desc">100% de origen vegetal. Legumbres, tofu y proteínas vegetales.</div>
        <div class="diet-macros">
          <div class="macro"><div class="macro-val">55%</div><div class="macro-label">Carbos</div></div>
          <div class="macro"><div class="macro-val">25%</div><div class="macro-label">Proteína</div></div>
          <div class="macro"><div class="macro-val">20%</div><div class="macro-label">Grasas</div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- NUTRICION DETAIL -->
  <div class="screen" id="screen-nutrition-detail">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Recomendaciones</span><span class="sep">›</span><span>Objetivo</span><span class="sep">›</span><span class="current">Dieta detallada</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('nutrition')">←</button>
      <div class="top-nav-title" id="diet-detail-title">Mediterránea</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Consultar dieta detallada</div>
      <div class="step-sub">Con alimentación sugerida para el día</div>

      <div style="background:var(--green); border-radius:var(--radius); padding:18px; margin-bottom:20px; box-shadow:var(--shadow-md);">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <div style="font-size:12px; color:var(--yellow); margin-bottom:4px;">Calorías diarias</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:24px; font-weight:700; color:white;" id="diet-detail-cal">1.800 kcal</div>
          </div>
          <div style="font-size:36px;" id="diet-detail-emoji">🫒</div>
        </div>
      </div>

      <div style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:700; color:var(--text); margin-bottom:12px;">Alimentación sugerida</div>

      <div class="meal-row">
        <div class="meal-time">🌅 Desayuno (07:30)</div>
        <div class="meal-desc">Avena con frutas y proteína en polvo · Tostada con palta y huevo</div>
      </div>
      <div class="meal-row">
        <div class="meal-time">🍎 Media mañana (10:30)</div>
        <div class="meal-desc">Yogur griego con almendras y miel</div>
      </div>
      <div class="meal-row">
        <div class="meal-time">☀️ Almuerzo (13:00)</div>
        <div class="meal-desc">Pollo grillado con quinoa y vegetales asados · Ensalada verde</div>
      </div>
      <div class="meal-row">
        <div class="meal-time">🥤 Merienda (17:00)</div>
        <div class="meal-desc">Batido proteico con banana y mantequilla de maní</div>
      </div>
      <div class="meal-row">
        <div class="meal-time">🌙 Cena (20:30)</div>
        <div class="meal-desc">Salmón con batata al horno y espinaca salteada</div>
      </div>

      <div style="margin-top:24px;">
        <button class="btn-primary" onclick="goto('home'); showAlert('Plan guardado', 'success')">Guardar este plan</button>
        <button class="btn-secondary" onclick="goto('nutrition')">Volver a recomendaciones</button>
      </div>
    </div>
  </div>

  <!-- PUNTOS VIDA CHECK -->
  <div class="screen" id="screen-puntos-vida-check">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span class="current">¿Ver saldo o beneficios?</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('home')">←</button>
      <div class="top-nav-title">Puntos Vida</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content" style="display:flex; flex-direction:column; justify-content:center;">
      <div class="step-title" style="text-align:center;">¿Qué querés hacer?</div>
      <div class="step-sub" style="text-align:center;">Elegí entre consultar tu saldo o canjear beneficios</div>

      <div class="choice-card" onclick="goto('puntos-saldo')">
        <div class="choice-card-icon">💰</div>
        <div class="choice-card-info">
          <div class="choice-card-title">Saldo</div>
          <div class="choice-card-sub">Consultar puntos y vía de obtención</div>
        </div>
        <div class="choice-card-arrow">→</div>
      </div>

      <div class="choice-card" onclick="goto('puntos-beneficios')">
        <div class="choice-card-icon">🎁</div>
        <div class="choice-card-info">
          <div class="choice-card-title">Beneficios</div>
          <div class="choice-card-sub">Canjear puntos por premios</div>
        </div>
        <div class="choice-card-arrow">→</div>
      </div>
    </div>
  </div>

  <!-- PUNTOS SALDO -->
  <div class="screen" id="screen-puntos-saldo">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Puntos Vida</span><span class="sep">›</span><span class="current">Consultar saldo + vía de obtención</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('puntos-vida-check')">←</button>
      <div class="top-nav-title">Saldo</div>
      <div style="width:36px;"></div>
    </div>

    <div class="points-hero">
      <div class="points-label">Saldo de puntos</div>
      <div class="points-value" id="display-points-saldo">2.450</div>
      <div class="points-sub">Equivalen a $2.450 en beneficios</div>
    </div>

    <div class="screen-content" style="padding-top:0;">
      <div style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:700; color:var(--text); margin-bottom:12px;">Vía de obtención (objetivos)</div>

      <div class="obtencion-list">
        <div class="obtencion-item">
          <div>
            <div class="obtencion-label">✅ Ir al gimnasio</div>
            <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">Por cada visita registrada</div>
          </div>
          <div class="obtencion-value">+10 pts</div>
        </div>
        <div class="obtencion-item">
          <div>
            <div class="obtencion-label">🎥 Completar clase virtual</div>
            <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">Por clase finalizada</div>
          </div>
          <div class="obtencion-value">+20 pts</div>
        </div>
        <div class="obtencion-item">
          <div>
            <div class="obtencion-label">💪 Completar rutina</div>
            <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">Por rutina finalizada</div>
          </div>
          <div class="obtencion-value">+15 pts</div>
        </div>
        <div class="obtencion-item">
          <div>
            <div class="obtencion-label">🎯 Cumplir objetivo semanal</div>
            <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">Por semana completa</div>
          </div>
          <div class="obtencion-value">+50 pts</div>
        </div>
        <div class="obtencion-item">
          <div>
            <div class="obtencion-label">👥 Referir un amigo</div>
            <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">Cuando se suscribe</div>
          </div>
          <div class="obtencion-value">+100 pts</div>
        </div>
      </div>

      <button class="btn-secondary" onclick="goto('puntos-beneficios')">Ver beneficios disponibles →</button>
    </div>
  </div>

  <!-- PUNTOS BENEFICIOS -->
  <div class="screen" id="screen-puntos-beneficios">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="points-saldo-corner" id="display-points-corner">2.450 pts</div>
    <div class="step-trail"><span>Puntos Vida</span><span class="sep">›</span><span class="current">Lista de beneficios</span><span class="sep">›</span><span>Seleccionar</span><span class="sep">›</span><span>¿Confirmar canje?</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('puntos-vida-check')">←</button>
      <div class="top-nav-title">Beneficios</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Ver lista de beneficios</div>
      <div class="step-sub">Seleccioná un beneficio para canjear</div>

      <div class="benefit-card" onclick="selectBeneficio('🧴 Proteína en polvo (1kg)', 1500)">
        <div class="benefit-icon-wrap">🧴</div>
        <div class="benefit-info">
          <div class="benefit-name">Proteína en polvo (1kg)</div>
          <div class="benefit-desc">Whey protein vainilla o chocolate</div>
        </div>
        <div class="benefit-cost">1.500 pts</div>
      </div>

      <div class="benefit-card" onclick="selectBeneficio('👕 Remera FitLife', 800)">
        <div class="benefit-icon-wrap">👕</div>
        <div class="benefit-info">
          <div class="benefit-name">Remera FitLife</div>
          <div class="benefit-desc">Talle a elección</div>
        </div>
        <div class="benefit-cost">800 pts</div>
      </div>

      <div class="benefit-card" onclick="selectBeneficio('🥤 Botella de agua', 400)">
        <div class="benefit-icon-wrap">🥤</div>
        <div class="benefit-info">
          <div class="benefit-name">Botella de agua</div>
          <div class="benefit-desc">Acero inoxidable 750ml</div>
        </div>
        <div class="benefit-cost">400 pts</div>
      </div>

      <div class="benefit-card" onclick="selectBeneficio('🎟️ 1 mes gratis', 5000)">
        <div class="benefit-icon-wrap">🎟️</div>
        <div class="benefit-info">
          <div class="benefit-name">1 mes gratis</div>
          <div class="benefit-desc">Extensión de suscripción</div>
        </div>
        <div class="benefit-cost">5.000 pts</div>
      </div>

      <div class="benefit-card" onclick="selectBeneficio('👟 Zapatillas recicladas', 4500)">
        <div class="benefit-icon-wrap">👟</div>
        <div class="benefit-info">
          <div class="benefit-name">Zapatillas recicladas</div>
          <div class="benefit-desc">Plástico oceánico reciclado</div>
        </div>
        <div class="benefit-cost">4.500 pts</div>
      </div>

      <div class="benefit-card" onclick="selectBeneficio('🎒 Mochila eco', 2200)">
        <div class="benefit-icon-wrap">🎒</div>
        <div class="benefit-info">
          <div class="benefit-name">Mochila eco</div>
          <div class="benefit-desc">Tela reciclada — diseño FitLife</div>
        </div>
        <div class="benefit-cost">2.200 pts</div>
      </div>

      <div class="benefit-card" onclick="selectBeneficio('👕 Remera algodón orgánico', 1000)">
        <div class="benefit-icon-wrap">👕</div>
        <div class="benefit-info">
          <div class="benefit-name">Remera algodón orgánico</div>
          <div class="benefit-desc">Cultivo sin químicos · talle a elección</div>
        </div>
        <div class="benefit-cost">1.000 pts</div>
      </div>
    </div>
  </div>

  <!-- CANJE CHECK -->
  <div class="screen" id="screen-canje-check">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Beneficios</span><span class="sep">›</span><span>Seleccionado</span><span class="sep">›</span><span class="current">¿Confirmar canje?</span></div>
    <div class="decision-screen">
      <div class="decision-diamond">
        <div class="decision-diamond-inner">
          <div class="q-icon">🎁</div>
          <div class="q-text">¿Confirmar<br>canje?</div>
        </div>
      </div>

      <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:10px; padding:16px; width:100%; box-shadow:var(--shadow-sm);">
        <div style="font-size:14px; color:var(--text); font-weight:700; margin-bottom:12px; text-align:center;" id="canje-item-name">Proteína en polvo (1kg)</div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
          <span style="color:var(--text-muted); font-size:13px;">Tus puntos</span>
          <span style="color:var(--green); font-weight:700;" id="canje-saldo">2.450 pts</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
          <span style="color:var(--text-muted); font-size:13px;">Costo</span>
          <span style="color:var(--text); font-weight:700;" id="canje-cost">- 1.500 pts</span>
        </div>
        <div style="height:1px; background:var(--border); margin:10px 0;"></div>
        <div style="display:flex; justify-content:space-between;">
          <span style="color:var(--text-muted); font-size:13px;">Quedarán</span>
          <span style="color:var(--green); font-weight:700;" id="canje-restante">950 pts</span>
        </div>
      </div>

      <div class="decision-actions">
        <button class="btn-danger" onclick="cancelarCanje()" style="margin-top:0;">No, cancelar</button>
        <button class="btn-primary" onclick="confirmarCanje()">Sí, confirmar</button>
      </div>
    </div>
  </div>

  <!-- CANJE EXITOSO -->
  <div class="screen" id="screen-canje-exitoso">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="granted-screen">
      <div class="granted-icon">🎉</div>
      <div class="granted-title">¡Canje exitoso!</div>
      <div class="granted-sub">Descontamos los puntos y recibirás tu beneficio en breve.</div>
      <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius-sm); padding:14px; margin-top:16px; width:100%; box-shadow:var(--shadow-sm);">
        <div style="font-size:13px; color:var(--text);" id="exito-canje-item">Proteína en polvo (1kg)</div>
        <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">Código: <span class="text-green" id="exito-canje-code">FL-2026-A4B7</span></div>
      </div>
      <button class="btn-primary" onclick="goto('home')" style="margin-top:16px;">Volver al inicio</button>
      <button class="btn-secondary" onclick="goto('puntos-beneficios')">Ver más beneficios</button>
    </div>
  </div>

  <!-- CLASSES -->
  <div class="screen" id="screen-classes">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span class="current">Ver catálogo</span><span class="sep">›</span><span>Filtrar</span><span class="sep">›</span><span>Seleccionar</span><span class="sep">›</span><span>¿Inscribirse?</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('home')">←</button>
      <div class="top-nav-title">Clases</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Ver catálogo de clases</div>
      <div class="step-sub">Virtuales o presenciales — Filtrá por categoría/nivel (opcional)</div>

      <div style="display:flex; gap:8px; margin-bottom:12px;">
        <button class="filter-chip active" onclick="filterTipo(this, 'todas')">Todas</button>
        <button class="filter-chip" onclick="filterTipo(this, 'virtual')">🎥 Virtuales</button>
        <button class="filter-chip" onclick="filterTipo(this, 'presencial')">📍 Presenciales</button>
      </div>

      <div class="filter-row">
        <button class="filter-chip active" onclick="filterCat(this)">Todas</button>
        <button class="filter-chip" onclick="filterCat(this)">Fuerza</button>
        <button class="filter-chip" onclick="filterCat(this)">Cardio</button>
        <button class="filter-chip" onclick="filterCat(this)">Yoga</button>
        <button class="filter-chip" onclick="filterCat(this)">HIIT</button>
        <button class="filter-chip" onclick="filterCat(this)">Principiante</button>
        <button class="filter-chip" onclick="filterCat(this)">Avanzado</button>
      </div>

      <div class="class-card" onclick="goto('class-detail', 'Full Body HIIT|HIIT|Virtual|45 min|Intermedio|🔥')">
        <div class="class-thumb" style="background: linear-gradient(135deg, #1A3A2A, #2D5A3D);">🔥
          <div class="class-tipo-badge">🎥 Virtual</div>
          <div class="class-badge">HIIT</div>
        </div>
        <div class="class-body">
          <div class="class-name">Full Body HIIT</div>
          <div class="class-meta">
            <div class="class-meta-item">⏱ 45 min</div>
            <div class="class-meta-item">📊 Intermedio</div>
            <div class="class-meta-item">👤 12 insc.</div>
          </div>
        </div>
      </div>

      <div class="class-card" onclick="goto('class-detail', 'Yoga Restore|Yoga|Presencial|60 min|Principiante|🧘')">
        <div class="class-thumb" style="background: linear-gradient(135deg, #2D5A3D, #4CAF6F);">🧘
          <div class="class-tipo-badge">📍 Presencial</div>
          <div class="class-badge">Yoga</div>
        </div>
        <div class="class-body">
          <div class="class-name">Yoga Restore</div>
          <div class="class-meta">
            <div class="class-meta-item">⏱ 60 min</div>
            <div class="class-meta-item">📊 Principiante</div>
            <div class="class-meta-item">👤 8 insc.</div>
          </div>
        </div>
      </div>

      <div class="class-card" onclick="goto('class-detail', 'Strength & Power|Fuerza|Presencial|50 min|Avanzado|💪')">
        <div class="class-thumb" style="background: linear-gradient(135deg, #1A3A2A, #4CAF6F);">💪
          <div class="class-tipo-badge">📍 Presencial</div>
          <div class="class-badge">Fuerza</div>
        </div>
        <div class="class-body">
          <div class="class-name">Strength & Power</div>
          <div class="class-meta">
            <div class="class-meta-item">⏱ 50 min</div>
            <div class="class-meta-item">📊 Avanzado</div>
            <div class="class-meta-item">👤 5 insc.</div>
          </div>
        </div>
      </div>

      <div class="class-card" onclick="goto('class-detail', 'Cardio Dance|Cardio|Virtual|40 min|Principiante|💃')">
        <div class="class-thumb" style="background: linear-gradient(135deg, #2D5A3D, #1A3A2A);">💃
          <div class="class-tipo-badge">🎥 Virtual</div>
          <div class="class-badge">Cardio</div>
        </div>
        <div class="class-body">
          <div class="class-name">Cardio Dance</div>
          <div class="class-meta">
            <div class="class-meta-item">⏱ 40 min</div>
            <div class="class-meta-item">📊 Principiante</div>
            <div class="class-meta-item">👤 20 insc.</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- CLASS DETAIL -->
  <div class="screen" id="screen-class-detail">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Catálogo</span><span class="sep">›</span><span class="current">Clase seleccionada</span><span class="sep">›</span><span>¿Inscribirse?</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('classes')">←</button>
      <div class="top-nav-title" id="cd-name">Full Body HIIT</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="class-detail-hero" id="cd-hero">🔥</div>

      <div class="class-detail-meta">
        <span class="yellow" id="cd-cat">HIIT</span>
        <span id="cd-tipo">🎥 Virtual</span>
        <span id="cd-dur">⏱ 45 min</span>
        <span id="cd-nivel">📊 Intermedio</span>
      </div>

      <p style="color:var(--text-soft); font-size:14px; line-height:1.6; margin-bottom:24px;">
        Clase de alta intensidad. Trabajo full body con intervalos cortos de máxima exigencia y descansos activos.
      </p>

      <button class="btn-primary" onclick="goto('inscribirse-check')">Continuar →</button>
    </div>
  </div>

  <!-- INSCRIBIRSE -->
  <div class="screen" id="screen-inscribirse-check">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Clase</span><span class="sep">›</span><span class="current">¿Inscribirse?</span></div>
    <div class="decision-screen">
      <div class="decision-diamond">
        <div class="decision-diamond-inner">
          <div class="q-icon">📝</div>
          <div class="q-text">¿Desea<br>inscribirse?</div>
        </div>
      </div>
      <div class="decision-context">¿Querés inscribirte a <span class="text-green" id="inscribirse-class-name">Full Body HIIT</span>?</div>
      <div class="decision-actions">
        <button class="btn-danger" onclick="goto('volver-salir-check')" style="margin-top:0;">No</button>
        <button class="btn-primary" onclick="goto('horario')">Sí, inscribirme</button>
      </div>
    </div>
  </div>

  <!-- HORARIO -->
  <div class="screen" id="screen-horario">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Inscribirse</span><span class="sep">›</span><span class="current">Elegir día y horario</span><span class="sep">›</span><span>Confirmar</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('inscribirse-check')">←</button>
      <div class="top-nav-title">Día y horario</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Elegir día y horario</div>
      <div class="step-sub">Disponibles para los próximos 7 días</div>

      <div class="horario-grid">
        <div class="horario-chip" onclick="selectHorarioChip(this)">Lun 07:00</div>
        <div class="horario-chip" onclick="selectHorarioChip(this)">Lun 19:00</div>
        <div class="horario-chip unavailable">Mar 07:00</div>
        <div class="horario-chip" onclick="selectHorarioChip(this)">Mar 19:00</div>
        <div class="horario-chip" onclick="selectHorarioChip(this)">Mié 08:00</div>
        <div class="horario-chip" onclick="selectHorarioChip(this)">Mié 19:00</div>
        <div class="horario-chip" onclick="selectHorarioChip(this)">Jue 07:00</div>
        <div class="horario-chip" onclick="selectHorarioChip(this)">Jue 19:00</div>
        <div class="horario-chip unavailable">Vie 07:00</div>
        <div class="horario-chip" onclick="selectHorarioChip(this)">Vie 19:00</div>
        <div class="horario-chip" onclick="selectHorarioChip(this)">Sáb 10:00</div>
        <div class="horario-chip" onclick="selectHorarioChip(this)">Sáb 11:00</div>
      </div>

      <button class="btn-primary" id="btn-ir-inscripcion" onclick="goto('proceso-inscripcion')" style="opacity:0.5; pointer-events:none;">Ir al proceso de inscripción →</button>
    </div>
  </div>

  <!-- PROCESO INSCRIPCION -->
  <div class="screen" id="screen-proceso-inscripcion">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Horario</span><span class="sep">›</span><span class="current">Confirmar reserva</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('horario')">←</button>
      <div class="top-nav-title">Confirmar reserva</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Proceso de inscripción</div>
      <div class="step-sub">Revisá los datos antes de confirmar</div>

      <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:18px; margin-bottom:20px; box-shadow:var(--shadow-sm);">
        <div style="font-size:12px; color:var(--text-muted); margin-bottom:4px;">Clase</div>
        <div style="font-size:16px; font-weight:700; color:var(--text); margin-bottom:12px;" id="conf-class">Full Body HIIT</div>

        <div style="font-size:12px; color:var(--text-muted); margin-bottom:4px;">Día y horario</div>
        <div style="font-size:16px; font-weight:700; color:var(--green); margin-bottom:12px;" id="conf-horario">Lun 19:00</div>

        <div style="font-size:12px; color:var(--text-muted); margin-bottom:4px;">Duración</div>
        <div style="font-size:14px; color:var(--text);" id="conf-dur">45 min</div>
      </div>

      <button class="btn-primary" onclick="confirmarReserva()">Confirmar reserva ✓</button>
      <button class="btn-secondary" onclick="goto('horario')">Cambiar horario</button>
    </div>
  </div>

  <!-- RESERVA CONFIRMADA -->
  <div class="screen" id="screen-reserva-confirmada">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="granted-screen">
      <div class="granted-icon">✓</div>
      <div class="granted-title">Reserva confirmada</div>
      <div class="granted-sub">Te enviamos los detalles de la inscripción a tu mail.</div>
      <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius-sm); padding:14px; margin-top:16px; width:100%; box-shadow:var(--shadow-sm);">
        <div style="font-size:13px; color:var(--text);" id="rc-class">Full Body HIIT</div>
        <div style="font-size:12px; color:var(--green); margin-top:4px; font-weight:700;" id="rc-horario">Lun 19:00</div>
      </div>
      <button class="btn-primary" onclick="goto('home')" style="margin-top:16px;">Volver al inicio</button>
    </div>
  </div>

  <!-- VOLVER SALIR -->
  <div class="screen" id="screen-volver-salir-check">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>No te inscribiste</span><span class="sep">›</span><span class="current">¿Volver al catálogo o salir?</span></div>
    <div class="decision-screen">
      <div class="decision-diamond">
        <div class="decision-diamond-inner">
          <div class="q-icon">↩️</div>
          <div class="q-text">¿Volver al<br>catálogo o<br>salir?</div>
        </div>
      </div>
      <div class="decision-context">Decidiste no inscribirte. ¿Qué querés hacer?</div>
      <div class="decision-actions">
        <button class="btn-secondary" onclick="goto('home')" style="margin-top:0;">Salir</button>
        <button class="btn-primary" onclick="goto('classes')">Volver al catálogo</button>
      </div>
    </div>
  </div>

  <!-- ROUTINES -->
  <div class="screen" id="screen-routines">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span class="current">Ver rutinas personalizadas</span><span class="sep">›</span><span>Elegir en base a objetivos</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('home')">←</button>
      <div class="top-nav-title">Rutinas</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Ver rutinas personalizadas</div>
      <div class="step-sub">Elegí en base a tus objetivos</div>

      <div class="objective-grid">
        <div class="objective-card" onclick="filterRoutine(this, 'perder')">
          <div class="obj-icon">🔥</div>
          <div class="obj-label">Perder peso</div>
        </div>
        <div class="objective-card selected" onclick="filterRoutine(this, 'musculo')">
          <div class="obj-icon">💪</div>
          <div class="obj-label">Ganar músculo</div>
        </div>
        <div class="objective-card" onclick="filterRoutine(this, 'rendimiento')">
          <div class="obj-icon">⚡</div>
          <div class="obj-label">Rendimiento</div>
        </div>
        <div class="objective-card" onclick="filterRoutine(this, 'flexibilidad')">
          <div class="obj-icon">🧘</div>
          <div class="obj-label">Flexibilidad</div>
        </div>
      </div>

      <div class="routine-card" onclick="showAlert('Rutina iniciada 💪', 'success')">
        <div class="routine-name">💪 Fuerza — Tren Superior</div>
        <div class="routine-obj">Ganar músculo</div>
        <div class="routine-exercises">Press de banca · Pull-ups · Remo · Curl de bíceps · 4 más</div>
        <div style="margin-top:10px; font-size:12px; color:var(--text-muted);">8 ejercicios · 45 min · Intermedio</div>
      </div>

      <div class="routine-card" onclick="showAlert('Rutina iniciada 🔥', 'success')">
        <div class="routine-name">🔥 HIIT — 20 min</div>
        <div class="routine-obj">Perder peso</div>
        <div class="routine-exercises">Burpees · Saltos · Mountain climbers · Sprint · 2 más</div>
        <div style="margin-top:10px; font-size:12px; color:var(--text-muted);">6 ejercicios · 20 min · Avanzado</div>
      </div>

      <div class="routine-card" onclick="showAlert('Rutina iniciada 🧘', 'success')">
        <div class="routine-name">🧘 Movilidad</div>
        <div class="routine-obj">Flexibilidad</div>
        <div class="routine-exercises">Estiramiento · Movilidad de cadera · Yoga flows · 4 más</div>
        <div style="margin-top:10px; font-size:12px; color:var(--text-muted);">7 ejercicios · 30 min · Principiante</div>
      </div>
    </div>
  </div>

  <!-- QR BIO -->
  <div class="screen" id="screen-qr-bio">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Generar QR</span><span class="sep">›</span><span class="current">Escaneo facial</span><span class="sep">›</span><span>QR</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('home')">←</button>
      <div class="top-nav-title">QR de acceso</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content" style="display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
      <div class="step-title">Verificación facial</div>
      <div class="step-sub">Escaneo de rostro requerido para generar QR</div>

      <div class="face-scan-frame" id="face-scan-frame">
        <div class="corner c-tl"></div>
        <div class="corner c-tr"></div>
        <div class="corner c-bl"></div>
        <div class="corner c-br"></div>
        <div class="face-emoji" id="face-emoji">😊</div>
      </div>

      <div id="bio-state-init">
        <p style="color:var(--text); font-size:15px; font-weight:600; margin-bottom:20px;">Posicioná tu rostro frente a la cámara</p>
        <button class="btn-primary" onclick="startBio()" style="max-width:280px;">Escanear rostro</button>
      </div>

      <div id="bio-state-checking" style="display:none;">
        <p style="color:var(--text-muted); font-size:14px;">Escaneando rostro...</p>
        <div class="spinner" style="margin: 16px auto; width:40px; height:40px; border-width:3px;"></div>
      </div>
    </div>
  </div>

  <!-- QR BIO RESULT -->
  <div class="screen" id="screen-qr-x-check">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Escaneo facial</span><span class="sep">›</span><span class="current">Resultado</span></div>
    <div class="decision-screen">
      <div class="decision-diamond">
        <div class="decision-diamond-inner">
          <div class="q-icon">🔍</div>
          <div class="q-text">¿Rostro<br>reconocido?</div>
        </div>
      </div>
      <div class="decision-context">Procesando reconocimiento facial...</div>
      <div class="spinner" style="margin: 0 auto;"></div>
    </div>
  </div>

  <!-- QR REINTENTAR -->
  <div class="screen" id="screen-qr-reintentar">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Escaneo fallido</span><span class="sep">›</span><span class="current">¿Reintentar?</span></div>
    <div class="decision-screen">
      <div class="decision-diamond" style="border-color:var(--red);">
        <div class="decision-diamond-inner">
          <div class="q-icon">❌</div>
          <div class="q-text">¿Reintentar?</div>
        </div>
      </div>
      <div class="decision-context">El escaneo facial falló. ¿Querés intentar de nuevo?</div>
      <div class="decision-actions">
        <button class="btn-danger" onclick="goto('home')" style="margin-top:0;">No</button>
        <button class="btn-primary" onclick="goto('qr-bio')">Sí, reintentar</button>
      </div>
    </div>
  </div>

  <!-- QR SHOW -->
  <div class="screen" id="screen-qr-show">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Bio exitosa</span><span class="sep">›</span><span class="current">Mostrar QR</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('home')">←</button>
      <div class="top-nav-title">QR de acceso</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content" style="text-align:center;">
      <div class="step-title" style="text-align:center;">Mostrar QR</div>
      <div class="step-sub" style="text-align:center;">Presentá este código en el lector de la sede</div>

      <div class="qr-box"><div class="qr-inner"></div></div>

      <div class="qr-timer">
        <div class="qr-timer-dot"></div>
        <span>Válido por <span id="qr-countdown">5:00</span></span>
      </div>

      <div class="qr-valid-badge">⏱ Válido por 5 minutos</div>

      <p style="color:var(--text-muted); font-size:12px; margin-top:24px;">El código expira automáticamente por seguridad</p>
    </div>
  </div>

  <!-- SEDES -->
  <div class="screen" id="screen-sedes">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span class="current">Ver mapa</span><span class="sep">›</span><span>Seleccionar sede</span><span class="sep">›</span><span>Consultar detalles</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('home')">←</button>
      <div class="top-nav-title">Sedes</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Ver mapa con sedes disponibles</div>
      <div class="step-sub">Seleccioná una sede para consultar detalles</div>

      <div class="map-placeholder">
        <span style="font-size:40px;">🗺️</span>
        <span style="color:var(--text-muted); font-size:13px;">Mapa interactivo de sedes</span>
      </div>

      <div class="sede-card" onclick="selectSede('Palermo', 'Honduras 4523', true)">
        <div class="sede-card-top">
          <div class="sede-name">FitLife Palermo</div>
          <div class="sede-open open">Abierto</div>
        </div>
        <div class="sede-address">📍 Honduras 4523, CABA</div>
        <div class="sede-info-row">
          <div class="sede-info-item">🕐 6:00 - 23:00</div>
          <div class="sede-info-item">👥 Media</div>
          <div class="sede-info-item">⭐ 4.8</div>
        </div>
      </div>

      <div class="sede-card" onclick="selectSede('Belgrano', 'Av. Monroe 2101', true)">
        <div class="sede-card-top">
          <div class="sede-name">FitLife Belgrano</div>
          <div class="sede-open open">Abierto</div>
        </div>
        <div class="sede-address">📍 Av. Monroe 2101, CABA</div>
        <div class="sede-info-row">
          <div class="sede-info-item">🕐 6:00 - 22:00</div>
          <div class="sede-info-item">👥 Baja</div>
          <div class="sede-info-item">⭐ 4.6</div>
        </div>
      </div>

      <div class="sede-card" onclick="selectSede('Microcentro', 'Av. Corrientes 1285', false)">
        <div class="sede-card-top">
          <div class="sede-name">FitLife Microcentro</div>
          <div class="sede-open closed">Cerrado</div>
        </div>
        <div class="sede-address">📍 Av. Corrientes 1285, CABA</div>
        <div class="sede-info-row">
          <div class="sede-info-item">🕐 7:00 - 21:00</div>
          <div class="sede-info-item">👥 —</div>
          <div class="sede-info-item">⭐ 4.5</div>
        </div>
      </div>
    </div>
  </div>

  <!-- SEDE DETAIL -->
  <div class="screen" id="screen-sede-detail">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Sede</span><span class="sep">›</span><span class="current">Consultar detalles</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('sedes')">←</button>
      <div class="top-nav-title" id="sd-name">FitLife Palermo</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Consultar detalles</div>
      <div class="step-sub" id="sd-address">📍 Honduras 4523, CABA</div>

      <div class="detail-option" onclick="goto('sede-direccion')">
        <div class="detail-option-icon">📍</div>
        <div class="detail-option-text">
          <div class="detail-option-title">Ver Dirección</div>
          <div class="detail-option-sub">Cómo llegar — Navegación GPS</div>
        </div>
        <span style="color:var(--green);">→</span>
      </div>

      <div class="detail-option" onclick="goto('sede-horarios')">
        <div class="detail-option-icon">🕐</div>
        <div class="detail-option-text">
          <div class="detail-option-title">Ver Horarios</div>
          <div class="detail-option-sub">Cuando la sede está abierta</div>
        </div>
        <span style="color:var(--green);">→</span>
      </div>

      <div class="detail-option" onclick="goto('sede-otros')">
        <div class="detail-option-icon">ℹ️</div>
        <div class="detail-option-text">
          <div class="detail-option-title">Ver Otros Detalles</div>
          <div class="detail-option-sub">Equipamiento, concurrencia, etc.</div>
        </div>
        <span style="color:var(--green);">→</span>
      </div>
    </div>
  </div>

  <!-- SEDE DIRECCION -->
  <div class="screen" id="screen-sede-direccion">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Detalles</span><span class="sep">›</span><span>Dirección</span><span class="sep">›</span><span class="current">¿Ir a la sede?</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('sede-detail')">←</button>
      <div class="top-nav-title">Ver Dirección</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Ver Dirección</div>

      <div class="map-placeholder" style="height:200px;">
        <span style="font-size:48px;">📍</span>
        <span style="color:var(--text-muted); font-size:13px;" id="sd-dir-text">Honduras 4523, CABA</span>
      </div>

      <div class="info-box">
        <div class="info-box-title">Dirección completa</div>
        <div class="info-box-content" id="sd-dir-full">Honduras 4523, Palermo, CABA, Argentina</div>
      </div>

      <div class="info-box">
        <div class="info-box-title">Cómo llegar</div>
        <div class="info-box-content">🚇 Subte D — Plaza Italia (5 min)<br>🚌 Colectivos 39, 152, 161<br>🚗 Estacionamiento gratuito</div>
      </div>

      <button class="btn-primary" onclick="goto('sede-ir-check')">¿Desea ir a la sede?</button>
    </div>
  </div>

  <!-- SEDE IR CHECK -->
  <div class="screen" id="screen-sede-ir-check">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Dirección</span><span class="sep">›</span><span class="current">¿Desea ir a la sede?</span></div>
    <div class="decision-screen">
      <div class="decision-diamond">
        <div class="decision-diamond-inner">
          <div class="q-icon">🗺️</div>
          <div class="q-text">¿Desea ir<br>a la sede?</div>
        </div>
      </div>
      <div class="decision-context">Si querés ir ahora, abrimos la navegación GPS</div>
      <div class="decision-actions">
        <button class="btn-secondary" onclick="goto('home')" style="margin-top:0;">No</button>
        <button class="btn-primary" onclick="goto('sede-gps')">Sí, abrir GPS</button>
      </div>
    </div>
  </div>

  <!-- SEDE GPS -->
  <div class="screen" id="screen-sede-gps">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="granted-screen">
      <div class="bio-icon-big">🗺️</div>
      <div class="granted-title">Abriendo GPS</div>
      <div class="granted-sub">Iniciando navegación a la sede seleccionada...</div>
      <div class="spinner" style="margin: 16px auto; width:40px; height:40px; border-width:3px;"></div>
    </div>
  </div>

  <!-- SEDE HORARIOS -->
  <div class="screen" id="screen-sede-horarios">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Detalles</span><span class="sep">›</span><span class="current">Ver Horarios</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('sede-detail')">←</button>
      <div class="top-nav-title">Ver Horarios</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Consultar horarios</div>
      <div class="step-sub">Cuando la sede está abierta</div>

      <div class="info-box">
        <div class="info-box-title">Lunes a Viernes</div>
        <div class="info-box-content">6:00 - 23:00</div>
      </div>
      <div class="info-box">
        <div class="info-box-title">Sábado</div>
        <div class="info-box-content">8:00 - 20:00</div>
      </div>
      <div class="info-box">
        <div class="info-box-title">Domingo</div>
        <div class="info-box-content">9:00 - 18:00</div>
      </div>
      <div class="info-box">
        <div class="info-box-title">Feriados</div>
        <div class="info-box-content">9:00 - 14:00 (consultar en redes)</div>
      </div>

      <button class="btn-secondary" onclick="goto('sede-detail')">Volver</button>
    </div>
  </div>

  <!-- SEDE OTROS -->
  <div class="screen" id="screen-sede-otros">
    <div class="status-bar"><span>9:41</span><div class="status-icons">▲ ● ▮▮▮</div></div>
    <div class="step-trail"><span>Detalles</span><span class="sep">›</span><span class="current">Otros Detalles</span></div>
    <div class="top-nav">
      <button class="back-btn" onclick="goto('sede-detail')">←</button>
      <div class="top-nav-title">Otros Detalles</div>
      <div style="width:36px;"></div>
    </div>
    <div class="screen-content">
      <div class="step-title">Consultar otros detalles</div>
      <div class="step-sub">Equipamiento, nivel de concurrencia, etc.</div>

      <div class="info-box">
        <div class="info-box-title">Equipamiento disponible</div>
        <div class="info-box-content">🏋️ 40 máquinas de fuerza<br>🚴 20 bicicletas estáticas<br>🏃 15 cintas de correr<br>🧘 Sala de clases grupales<br>🏊 Pileta climatizada</div>
      </div>

      <div class="info-box">
        <div class="info-box-title">Nivel de concurrencia (en vivo)</div>
        <div class="info-box-content">👥 Afluencia media — 62% de capacidad<br><span style="color:var(--green); font-size:13px; font-weight:700;">Buen momento para entrenar</span></div>
      </div>

      <div class="info-box">
        <div class="info-box-title">Servicios</div>
        <div class="info-box-content">🚿 Vestuarios con duchas<br>🔒 Lockers personales<br>☕ Bar / Cafetería<br>📶 WiFi gratuito<br>🅿️ Estacionamiento</div>
      </div>

      <div class="info-box">
        <div class="info-box-title">Calificación de usuarios</div>
        <div class="info-box-content"><span class="text-green">⭐⭐⭐⭐⭐ 4.8 / 5</span> · 342 reseñas</div>
      </div>

      <button class="btn-secondary" onclick="goto('sede-detail')">Volver</button>
    </div>
  </div>

</div>

<script>
  function goto(screenId, param) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    const target = document.getElementById('screen-' + screenId);
    if (target) {
      target.classList.add('active');
      target.scrollTop = 0;
    }

    if (screenId === 'cuota-check') { setTimeout(checkCuota, 1500); }
    if (screenId === 'send-mail') { setTimeout(() => goto('confirm-mail'), 1800); }
    if (screenId === 'processing') { setTimeout(() => goto('pago-exitoso-check'), 2000); }
    if (screenId === 'pago-exitoso-check') { setTimeout(checkPagoExitoso, 1800); }
    if (screenId === 'access-granted') { setTimeout(() => goto('home'), 1800); }
    if (screenId === 'home') { _registering = false; }
    if (screenId === 'qr-x-check') { setTimeout(checkBioResult, 1500); }
    if (screenId === 'qr-show') { startQRTimer(); }
    if (screenId === 'sede-gps') { setTimeout(() => { showAlert('GPS iniciado 🗺️', 'success'); goto('home'); }, 1800); }
    if (screenId === 'nutrition-detail' && param) { setDietDetail(param); }
    if (screenId === 'class-detail' && param) { setClassDetail(param); }
  }

  function doLogin() {
    const email = document.getElementById('login-email').value;
    const pass = document.getElementById('login-pass').value;
    if (!email || !pass) { showAlert('Completá todos los campos', 'error'); return; }
    window._loginEmail = email;
    goto('cuota-check');
  }

  function checkCuota() {
    const tieneCuota = window._loginEmail !== 'vencida@test.com';
    if (tieneCuota) {
      document.getElementById('cuota-result').innerHTML = `
        <div style="background:#F0F8F2; border:1px solid var(--green); border-radius:var(--radius-sm); padding:14px; text-align:center;">
          <div style="color:var(--green); font-weight:700; margin-bottom:4px;">✓ Cuota al día</div>
          <div style="font-size:12px; color:var(--text-muted);">Acceso concedido — redirigiendo...</div>
        </div>
      `;
      setTimeout(() => goto('access-granted'), 1200);
    } else {
      document.getElementById('cuota-result').innerHTML = `
        <div style="background:rgba(224,72,72,0.08); border:1px solid var(--red); border-radius:var(--radius-sm); padding:14px; text-align:center;">
          <div style="color:var(--red); font-weight:700; margin-bottom:4px;">✗ Cuota vencida</div>
          <div style="font-size:12px; color:var(--text-muted);">Continuando...</div>
        </div>
      `;
      setTimeout(() => goto('renovar-check'), 1200);
    }
  }

  function doRegister() {
    const name = document.getElementById('reg-name').value;
    const email = document.getElementById('reg-email').value;
    const pass = document.getElementById('reg-pass').value;
    if (!name || !email || !pass) { showAlert('Completá todos los campos', 'error'); return; }
    document.getElementById('sent-mail-addr').textContent = email;
    document.getElementById('confirm-mail-addr').textContent = email;
    const nameParts = name.trim().split(' ');
    document.getElementById('home-username').innerHTML = nameParts[0] + (nameParts[1] ? ' <span>' + nameParts[1] + '</span>' : '');
    document.getElementById('perfil-name').textContent = name;
    document.getElementById('perfil-mail').textContent = email;
    goto('send-mail');
  }

  let _registering = false;
  function doSignTerms() {
    if (!document.getElementById('terms-accept').checked) {
      showAlert('Debés aceptar los términos', 'error');
      return;
    }
    _registering = true;
    goto('plan');
  }

  let _selectedPlan = null;
  function selectPlan(el, plan, price) {
    document.querySelectorAll('.plan-card').forEach(c => c.classList.remove('selected'));
    el.classList.add('selected');
    _selectedPlan = { plan, price };
    document.getElementById('pagar-plan-label').textContent = plan;
    document.getElementById('payment-plan-label').textContent = plan + ' — ' + price + '/mes';
    document.getElementById('home-plan-label').textContent = 'Plan ' + plan + ' activo';
    document.getElementById('perfil-plan-badge').textContent = '⭐ Plan ' + plan;
    const btn = document.getElementById('btn-continue-plan');
    btn.style.opacity = '1';
    btn.style.pointerEvents = 'auto';
  }

  let _lastPaymentMethod = '';
  function procesarPago(metodo) {
    _lastPaymentMethod = metodo;
    document.getElementById('processing-method-label').textContent = metodo;
    goto('processing');
  }

  function elegirMetodo(metodo) {
    _lastPaymentMethod = metodo;
    document.getElementById('card-form-method').textContent = metodo;
    goto('card-form');
  }

  function formatCardNumber(el) {
    let v = el.value.replace(/\D/g, '').substring(0, 16);
    el.value = v.replace(/(\d{4})(?=\d)/g, '$1 ');
    updatePreview();
  }

  function formatExp(el) {
    let v = el.value.replace(/\D/g, '').substring(0, 4);
    if (v.length >= 3) v = v.substring(0, 2) + '/' + v.substring(2);
    el.value = v;
    updatePreview();
  }

  function updatePreview() {
    const num = document.getElementById('card-number').value;
    const name = document.getElementById('card-name').value;
    const exp = document.getElementById('card-exp').value;
    document.getElementById('prev-number').textContent = num || '•••• •••• •••• ••••';
    document.getElementById('prev-name').textContent = (name || 'NOMBRE APELLIDO').toUpperCase();
    document.getElementById('prev-exp').textContent = exp || 'MM/AA';
  }

  function submitCard() {
    const num = document.getElementById('card-number').value.replace(/\s/g, '');
    const name = document.getElementById('card-name').value;
    const exp = document.getElementById('card-exp').value;
    const cvv = document.getElementById('card-cvv').value;
    if (num.length < 13 || !name || exp.length < 5 || cvv.length < 3) {
      showAlert('Completá todos los datos', 'error');
      return;
    }
    document.getElementById('processing-method-label').textContent = _lastPaymentMethod;
    goto('processing');
  }

  function checkPagoExitoso() {
    const exitoso = _lastPaymentMethod !== 'MercadoPago';
    if (exitoso) { goto('access-granted'); } else { goto('reintentar-check'); }
  }

  function selectNutObj(el, key, label) {
    document.querySelectorAll('#screen-nutrition .objective-card').forEach(c => c.classList.remove('selected'));
    el.classList.add('selected');
    showAlert('Objetivo: ' + label, 'success');
  }

  function setDietDetail(name) {
    const data = {
      'Mediterránea': { cal: '1.800 kcal', emoji: '🫒' },
      'Alta en proteínas': { cal: '2.200 kcal', emoji: '🥩' },
      'Plant-based': { cal: '1.600 kcal', emoji: '🌱' }
    };
    const d = data[name] || data['Mediterránea'];
    document.getElementById('diet-detail-title').textContent = name;
    document.getElementById('diet-detail-cal').textContent = d.cal;
    document.getElementById('diet-detail-emoji').textContent = d.emoji;
  }

  let currentPoints = 2450;
  let _canjeItem = null;
  let _canjeCost = 0;

  function selectBeneficio(name, cost) {
    _canjeItem = name;
    _canjeCost = cost;
    document.getElementById('canje-item-name').textContent = name;
    document.getElementById('canje-saldo').textContent = currentPoints.toLocaleString() + ' pts';
    document.getElementById('canje-cost').textContent = '- ' + cost.toLocaleString() + ' pts';
    document.getElementById('canje-restante').textContent = (currentPoints - cost).toLocaleString() + ' pts';
    if (currentPoints < cost) { showAlert('No tenés suficientes puntos', 'error'); return; }
    goto('canje-check');
  }

  function confirmarCanje() {
    currentPoints -= _canjeCost;
    document.getElementById('display-points-saldo').textContent = currentPoints.toLocaleString();
    document.getElementById('display-points-corner').textContent = currentPoints.toLocaleString() + ' pts';
    document.getElementById('home-points').textContent = currentPoints.toLocaleString();
    document.getElementById('exito-canje-item').textContent = _canjeItem;
    document.getElementById('exito-canje-code').textContent = 'FL-2026-' + Math.random().toString(36).substring(2,6).toUpperCase();
    goto('canje-exitoso');
  }

  function cancelarCanje() {
    showAlert('Canje cancelado', 'error');
    goto('puntos-beneficios');
  }

  function filterTipo(btn, tipo) {
    btn.parentElement.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
  }
  function filterCat(btn) {
    document.querySelectorAll('.filter-row .filter-chip').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
  }

  function setClassDetail(param) {
    const [name, cat, tipo, dur, nivel, emoji] = param.split('|');
    document.getElementById('cd-name').textContent = name;
    document.getElementById('cd-cat').textContent = cat;
    document.getElementById('cd-tipo').textContent = tipo === 'Virtual' ? '🎥 Virtual' : '📍 Presencial';
    document.getElementById('cd-dur').textContent = '⏱ ' + dur;
    document.getElementById('cd-nivel').textContent = '📊 ' + nivel;
    document.getElementById('cd-hero').textContent = emoji;
    document.getElementById('inscribirse-class-name').textContent = name;
    document.getElementById('conf-class').textContent = name;
    document.getElementById('conf-dur').textContent = dur;
    document.getElementById('rc-class').textContent = name;
  }

  let _selectedHorario = '';
  function selectHorarioChip(el) {
    document.querySelectorAll('.horario-chip').forEach(c => c.classList.remove('selected'));
    el.classList.add('selected');
    _selectedHorario = el.textContent;
    document.getElementById('conf-horario').textContent = _selectedHorario;
    document.getElementById('rc-horario').textContent = _selectedHorario;
    const btn = document.getElementById('btn-ir-inscripcion');
    btn.style.opacity = '1';
    btn.style.pointerEvents = 'auto';
  }

  function confirmarReserva() { goto('reserva-confirmada'); }

  function filterRoutine(el, key) {
    document.querySelectorAll('#screen-routines .objective-card').forEach(c => c.classList.remove('selected'));
    el.classList.add('selected');
  }

  let qrTimer = null;
  let _scanLineEl = null;

  function startBio() {
    document.getElementById('bio-state-init').style.display = 'none';
    document.getElementById('bio-state-checking').style.display = 'block';
    document.getElementById('face-emoji').textContent = '📷';
    const frame = document.getElementById('face-scan-frame');
    if (!_scanLineEl) {
      _scanLineEl = document.createElement('div');
      _scanLineEl.className = 'face-scan-line';
      frame.appendChild(_scanLineEl);
    }
    setTimeout(() => { goto('qr-x-check'); }, 2200);
  }

  function checkBioResult() {
    const success = Math.random() > 0.3;
    if (success) { goto('qr-show'); } else { goto('qr-reintentar'); }
    document.getElementById('bio-state-init').style.display = 'block';
    document.getElementById('bio-state-checking').style.display = 'none';
    document.getElementById('face-emoji').textContent = '😊';
    if (_scanLineEl && _scanLineEl.parentNode) {
      _scanLineEl.parentNode.removeChild(_scanLineEl);
      _scanLineEl = null;
    }
  }

  function startQRTimer() {
    let seconds = 300;
    clearInterval(qrTimer);
    document.getElementById('qr-countdown').textContent = '5:00';
    qrTimer = setInterval(() => {
      seconds--;
      const m = Math.floor(seconds / 60);
      const s = seconds % 60;
      document.getElementById('qr-countdown').textContent = m + ':' + (s < 10 ? '0' : '') + s;
      if (seconds <= 0) {
        clearInterval(qrTimer);
        showAlert('QR expirado', 'error');
        goto('qr-bio');
      }
    }, 1000);
  }

  function selectSede(name, address, isOpen) {
    document.getElementById('sd-name').textContent = 'FitLife ' + name;
    document.getElementById('sd-address').textContent = '📍 ' + address + ', CABA';
    document.getElementById('sd-dir-text').textContent = address + ', CABA';
    document.getElementById('sd-dir-full').textContent = address + ', ' + name + ', CABA, Argentina';
    goto('sede-detail');
  }

  function showAlert(msg, type) {
    const alert = document.getElementById('global-alert');
    alert.textContent = (type === 'success' ? '✓ ' : '✗ ') + msg;
    alert.className = 'alert show ' + type;
    setTimeout(() => { alert.className = 'alert'; }, 2500);
  }

  setTimeout(() => goto('auth-choice'), 2000);
</script>
</body>
</html>
