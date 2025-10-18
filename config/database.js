import mongoose from 'mongoose';
import dotenv from 'dotenv';
dotenv.config();

/**
 * Connect to MongoDB database
 * This function handles the connection and provides error handling
 */
const connectDB = async () => {
  try {
    // Connect to MongoDB using the URI from environment variables
    const conn = await mongoose.connect(process.env.MONGO_URI);

    console.log(`✅ MongoDB Connected: ${conn.connection.host}`);
    
    // Log the database name
    console.log(`📁 Database: ${conn.connection.name}`);
    
  } catch (error) {
    console.error(`❌ Error connecting to MongoDB: ${error.message}`);
    // Exit the process with failure (INSIDE catch block!)
    process.exit(1);
  }
};

// Handle connection events for better monitoring
mongoose.connection.on('disconnected', () => {
  console.log('⚠️  MongoDB disconnected');
});

mongoose.connection.on('error', (err) => {
  console.error(`❌ MongoDB connection error: ${err}`);
});

export default connectDB;