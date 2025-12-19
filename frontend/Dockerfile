# frontend/Dockerfile

# --- Stage 1: Build the React App ---
FROM node:18-alpine as build

WORKDIR /app

# Install dependencies (caches this layer if package.json doesn't change)
COPY package*.json ./
RUN npm install

# Copy source and build
COPY . .
RUN npm run build

# --- Stage 2: Serve with Nginx ---
FROM nginx:alpine

# Copy the built files from Stage 1 to Nginx's serve folder
COPY --from=build /app/dist /usr/share/nginx/html

# Expose HTTP port
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]