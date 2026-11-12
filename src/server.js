/**
 * VoxASR Render Production Server Entrypoint
 * Binds Hono application to 0.0.0.0:$PORT with graceful shutdown & DB/KB bootstrap.
 */

import { serve } from '@hono/node-server'
import app from './index.js'
import { closeDatabase, initDatabase } from './db.js'
import { reloadModel } from './inference.js'
import { initKnowledgeEngine } from './web_learner.js'

const PORT = Number(process.env.PORT || 10000)
const HOST = process.env.HOST || '0.0.0.0'

async function start() {
  await initDatabase()
  await initKnowledgeEngine(reloadModel)

  const server = serve(
    {
      fetch: app.fetch,
      port: PORT,
      hostname: HOST
    },
    (info) => {
      console.log(`[VoxASR] Listening on http://${HOST}:${info.port}`)
    }
  )

  let shuttingDown = false
  const shutdown = async (signal) => {
    if (shuttingDown) return
    shuttingDown = true
    console.log(`[VoxASR] Received ${signal}, shutting down gracefully...`)
    server.close(async () => {
      await closeDatabase()
      console.log('[VoxASR] Clean shutdown complete.')
      process.exit(0)
    })
    setTimeout(() => process.exit(1), 15000).unref()
  }

  process.on('SIGTERM', () => shutdown('SIGTERM'))
  process.on('SIGINT', () => shutdown('SIGINT'))
}

start().catch((err) => {
  console.error('[VoxASR] Fatal startup error:', err)
  process.exit(1)
})
