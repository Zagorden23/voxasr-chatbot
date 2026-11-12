module.exports = {
  apps: [
    {
      name: 'webapp',
      script: 'npx',
      args: 'wrangler pages dev dist --d1=webapp-production --local --ip 0.0.0.0 --port 3000',
      cwd: 'D:/chatbot_ASR/webapp',
      env: {
        NODE_ENV: 'development',
        PORT: 3000
      },
      watch: false,
      instances: 1,
      exec_mode: 'fork'
    },
    {
      name: 'knowledge-daemon',
      script: 'D:/chatbot_ASR/.venv/Scripts/python.exe',
      args: 'training/upgrade_server.py',
      cwd: 'D:/chatbot_ASR/webapp',
      env: {
        PORT: 3001
      },
      watch: false,
      instances: 1,
      exec_mode: 'fork'
    }
  ]
}
