import mongoose from "mongoose";

// connect to mongodb 

export default connectDB = async () => {
    try{
        const conn = await mongoose.connect(process.env.MONGO_URI);

        console.log(`Connection to MongoDB was a success!`)
    }
    catch (error){

        console.error(error);

    }
    process.exit(1)
};

// handle connection events 

mongoose.connection.on('disconnected', () => {
    console.log("MongoDB was disconnected");
});
mongoose.connection.on('error', (err) => {
    console.error(err)
});

