# Description: Dockerfile for the frontend of the project
FROM node:20.11-alpine3.19 as build

# Set the working directory in the container
WORKDIR /app

# Copy the package.json and package-lock.json files to the container
COPY package*.json ./

# Clean install a project
RUN npm ci

# Copy the entire project directory to the container
COPY . .

# Build the project
RUN npm run build

# Stage 2
FROM nginx:alpine

# Copy the nginx configuration file to the container
ADD ./config/nginx.conf /etc/nginx/conf.d/default.conf

# Copy the build output to the nginx public folder
COPY --from=build /app/dist/frontend/browser /var/www/app/

# Expose port 80
EXPOSE 80

# Set the command to run the nginx server
CMD ["nginx", "-g", "daemon off;"]

