// Load environment variables first
import dotenv from 'dotenv'
dotenv.config();

import express from 'express';
import  cors from 'cors';
import connectDB from './config/database.js';
import markdownRoutes from './routes/markdownRoutes.js';

// Initialize Express app
const app = express();

// Connect to MongoDB
connectDB();

// Middleware
// =====================================

// 1. CORS - Allow requests from React frontend
app.use(cors({
  origin: process.env.CLIENT_URL || 'http://localhost:3000',
  credentials: true
}));

// 2. Body Parser - Parse JSON request bodies
app.use(express.json());

// 3. Body Parser - Parse URL-encoded bodies
app.use(express.urlencoded({ extended: true }));

// Request logging middleware (helpful for debugging)
app.use((req, res, next) => {
  console.log(`${req.method} ${req.path} - ${new Date().toISOString()}`);
  next();
});

// Routes
// =====================================

// Health check endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'Markdown API Server is running! 🚀',
    version: '1.0.0',
    endpoints: {
      getAllDocs: 'GET /api/markdown',
      getOneDoc: 'GET /api/markdown/:id',
      createDoc: 'POST /api/markdown',
      updateDoc: 'PUT /api/markdown/:id',
      deleteDoc: 'DELETE /api/markdown/:id'
    }
  });
});

// API routes
app.use('/api/markdown', markdownRoutes);

// 404 Handler - Catch undefined routes
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: 'Route not found',
    path: req.path
  });
});

// Global Error Handler
app.use((err, req, res, next) => {
  console.error('Global error handler:', err);
  res.status(err.status || 500).json({
    success: false,
    message: err.message || 'Internal server error',
    error: process.env.NODE_ENV === 'development' ? err : {}
  });
});

// Start Server
// =====================================
const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════╗
║  🚀 Server running on port ${PORT}      ║
║  📝 Markdown API Server Active        ║
║  🌐 http://localhost:${PORT}            ║
╚════════════════════════════════════════╝
  `);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (err) => {
  console.error('❌ Unhandled Rejection:', err);
  // Close server & exit process
  server.close(() => process.exit(1));
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('👋 SIGTERM received. Shutting down gracefully...');
  server.close(() => {
    console.log('✅ Process terminated');
  });
});